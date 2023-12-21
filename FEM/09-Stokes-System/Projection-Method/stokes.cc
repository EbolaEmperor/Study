//----------------------------------------------------------
// This code is for stokes system
// See section 6.2 of Zhang [2016]
// 
// Method: Leray-Helmoltz projection with multigrid
// Space discretization: Q2 element (2nd order in H1)
//----------------------------------------------------------

#include <omp.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <iostream>
#include <chrono>

struct CPUTimer
{
  using HRC = std::chrono::high_resolution_clock;
  std::chrono::time_point<HRC>  start;
  CPUTimer() { reset(); }
  void reset() { start = HRC::now(); }
  double operator() () const {
    std::chrono::duration<double> e = HRC::now() - start;
    return e.count();
  }
};

using namespace dealii;

#define NO_FORCING_TERM
#define OUTPUT_CG_ITERATIONS

const int n_thr = 6;

//--------------------------Data Structures for MG--------------------------


template <int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim>         &mapping,
              const FiniteElement<dim>   &fe,
              const unsigned              q_dgree,
              const UpdateFlags           upd_flags)
    : fe_values(mapping, fe, QGauss<dim>(q_dgree), upd_flags)
  {}

  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
  {}

  FEValues<dim> fe_values;
};


struct CopyData
{
  unsigned                                 level;
  FullMatrix<double>                       cell_matrix;
  Vector<double>                           cell_rhs;
  std::vector<types::global_dof_index>     local_dof_indices;

  template <class Iterator>
  void reinit(const Iterator &cell, unsigned dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);

    local_dof_indices.resize(dofs_per_cell);
    cell->get_active_or_mg_dof_indices(local_dof_indices);
    level = cell->level();
  }
};


//----------------------------Initial values------------------------------

template <int dim>
class Initial1 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
class Initial2 : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};


template <int dim>
double Initial1<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return M_PI*sin(M_PI*p[0])*sin(M_PI*p[1]) 
         - 2*M_PI*M_PI*M_PI*cos(2*M_PI*p[0])*sin(2*M_PI*p[1]);
}


template <int dim>
double Initial2<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return -M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]) 
         + 2*M_PI*M_PI*M_PI*sin(2*M_PI*p[0])*cos(2*M_PI*p[1]);
}


//---------------------------------Solver Class-----------------------------------


template <int dim>
class Stokes{
public:
  Stokes(const int &,
         const int &proj_steps=20);
  void run();

private:
  template <class Iterator>
  void cell_worker(const Iterator    &cell,
                   ScratchData<dim>  &scratch_data,
                   CopyData          &copy_data);
  void assemble_multigrid(const bool &dirichlet=false);

  void make_mesh();
  void setup_system();
  void setup_grad_pressure();
  void solve(Vector<double>& solution, const bool ispressure = false);
  void compute_divergence();
  void output_result();
  
  void pre_projections(Vector<double>& u1, 
                       Vector<double>& u2, 
                       const int &times);
  void projection_poisson_setup(const Vector<double>& u1, 
                                const Vector<double>& u2);

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints_u1;
  AffineConstraints<double> constraints_u2;
  AffineConstraints<double> constraints_pressure;

  SparsityPattern      sparsity_pattern;
  SparsityPattern      sparsity_pattern_pressure;

  SparseMatrix<double> laplace_matrix_pressure;
  SparseMatrix<double> system_matrix_pressure;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  MGLevelObject<SparsityPattern> mg_sparsity_patterns;
  MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;
  MGTransferPrebuilt<Vector<double>>  mg_transfer;

  Vector<double> grad_pressure_u1;
  Vector<double> grad_pressure_u2;

  Vector<double> pressure;
  Vector<double> solution_u1;
  Vector<double> solution_u2;
  Vector<double> divergence;
  Vector<double> system_rhs;

  int level;
  bool next_step;
  int proj_steps;
};


//---------------------------------Solver Implemention-----------------------------------


template <int dim>
Stokes<dim>::Stokes(const int &N, const int &proj_steps)
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe(2)
  , dof_handler(triangulation)
  , level(N)
  , next_step(false)
  , proj_steps(proj_steps)
{}


template <int dim>
void Stokes<dim>::make_mesh(){
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(level);
}


template <int dim>
void Stokes<dim>::output_result()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u1, "solution_u1");
  data_out.add_data_vector(solution_u2, "solution_u2");
  data_out.add_data_vector(pressure, "pressure");
  compute_divergence();
  data_out.add_data_vector(divergence, "div_u");

  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}


template <int dim>
void Stokes<dim>::solve(Vector<double>& solution, const bool ispressure)
{
  SolverControl            solver_control(2000, ispressure ? 1e-8 : 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  if(!ispressure && !next_step)
  {
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  } else {
    SolverControl coarse_solver_control(5000, 1e-9, false, false);
    SolverCG<Vector<double>> coarse_solver(coarse_solver_control);
    PreconditionIdentity id;
    MGCoarseGridIterativeSolver<Vector<double>,
                                SolverCG<Vector<double>>,
                                SparseMatrix<double>,
                                PreconditionIdentity>
        coarse_grid_solver(coarse_solver, mg_matrices[0], id);

    using Smoother = PreconditionChebyshev<SparseMatrix<double>, Vector<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(1);
    mg_smoother.set_symmetric(true);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);

    Multigrid<Vector<double>> mg(
      mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG< dim, Vector<double>, MGTransferPrebuilt<Vector<double>> >
      preconditioner(dof_handler, mg, mg_transfer);
    
    if(ispressure)
      solver.solve(system_matrix_pressure, solution, system_rhs, preconditioner);
    else
      solver.solve(system_matrix, solution, system_rhs, preconditioner);

    const double mean_value = VectorTools::compute_mean_value(
      dof_handler, QGauss<dim>(fe.degree + 1), solution, 0);
    solution.add(-mean_value);
  }

#ifdef OUTPUT_CG_ITERATIONS
  std::cout << "   " << solver_control.last_step()
            << " CG iterations." << std::endl;
#endif
}


template <int dim>
void Stokes<dim>::compute_divergence()
{
  Assert(dim == 2, ExcNotImplemented());
  system_matrix.copy_from(mass_matrix);
  system_rhs.reinit(solution_u1.size());

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_val_vel(fe,
                            quadrature_formula,
                            update_gradients | update_JxW_values |
                              update_values);
  const unsigned dpc = fe.n_dofs_per_cell(),
                 nqp = quadrature_formula.size();

  std::vector<types::global_dof_index> ldi(dpc);
  Vector<double>                       loc_rot(dpc);

  std::vector<Tensor<1, dim>> grad_u1(nqp), grad_u2(nqp);
  divergence = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_val_vel.reinit(cell);
      cell->get_dof_indices(ldi);
      fe_val_vel.get_function_gradients(solution_u1, grad_u1);
      fe_val_vel.get_function_gradients(solution_u2, grad_u2);
      loc_rot = 0.;
      for (unsigned int q = 0; q < nqp; ++q)
        for (unsigned int i = 0; i < dpc; ++i)
          loc_rot(i) += (grad_u1[q][0] + grad_u2[q][1]) * //
                        fe_val_vel.shape_value(i, q) *    //
                        fe_val_vel.JxW(q);

      for (unsigned int i = 0; i < dpc; ++i)
        system_rhs(ldi[i]) += loc_rot(i);
    }

  solve(divergence);
}


template <int dim>
void Stokes<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::cout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  constraints_u1.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u1);
  constraints_u1.close();

  constraints_u2.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  constraints_u2.close();

  constraints_pressure.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_pressure);
  constraints_pressure.close();

  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_u1);
    sparsity_pattern.copy_from(dsp);
  }
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_pressure);
    sparsity_pattern_pressure.copy_from(dsp);
  }

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  laplace_matrix_pressure.reinit(sparsity_pattern_pressure);
  system_matrix_pressure.reinit(sparsity_pattern_pressure);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix_pressure);

  pressure.reinit(dof_handler.n_dofs());
  solution_u1.reinit(dof_handler.n_dofs());
  solution_u2.reinit(dof_handler.n_dofs());
  divergence.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  assemble_multigrid();
}


template <int dim>
template <class Iterator>
void Stokes<dim>::cell_worker(const Iterator     &cell,
                            ScratchData<dim>   &scratch_data,
                            CopyData           &copy_data)
{
  FEValues<dim> &fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  const unsigned dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
  copy_data.reinit(cell, dofs_per_cell);

  for(const unsigned q : fe_values.quadrature_point_indices())
  {
    const double weight = fe_values.JxW(q);
    for(unsigned i = 0; i < dofs_per_cell; ++i)
      for(unsigned j = 0; j < dofs_per_cell; ++j)
      {
        copy_data.cell_matrix(i,j) += fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q) *
                                      weight;
      }
    }
}


template <int dim>
void Stokes<dim>::assemble_multigrid(const bool &dirichlet)
{
  if(dirichlet) next_step = true;

  dof_handler.distribute_mg_dofs();
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);

  if(dirichlet){
    std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                      dirichlet_boundary_ids);
  }
  
  const unsigned n_levels = triangulation.n_levels();

  mg_interface_matrices.resize(0, n_levels-1);
  mg_matrices.resize(0, n_levels-1);
  mg_sparsity_patterns.resize(0, n_levels-1);
  mg_interface_sparsity_patterns.resize(0, n_levels-1);

  for(unsigned level = 0; level < n_levels; ++level)
  {
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(dof_handler, dsp, level);
      mg_sparsity_patterns[level].copy_from(dsp);
      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
    }
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level));
      MGTools::make_interface_sparsity_pattern(dof_handler,
                                               mg_constrained_dofs,
                                               dsp,
                                               level);
      mg_interface_sparsity_patterns[level].copy_from(dsp);
      mg_interface_matrices[level].reinit(mg_interface_sparsity_patterns[level]);
    }
  }

  MappingQ<dim> mapping(fe.degree);

  std::vector<AffineConstraints<double>> boundary_constraints(n_levels);
  for (unsigned int level = 0; level < n_levels; ++level)
    {
      const IndexSet dofset =
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
      boundary_constraints[level].reinit(dofset);
      boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));
      if(dirichlet){
        boundary_constraints[level].add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
      }
      boundary_constraints[level].close();
    }

  auto cell_worker = 
   [&] (const typename DoFHandler<dim>::level_cell_iterator  &cell,
        ScratchData<dim>                                     &scratch_data,
        CopyData                                             &copy_data) {
      this->cell_worker(cell, scratch_data, copy_data);
    };

  auto copier = [&](const CopyData &cd){
    boundary_constraints[cd.level].distribute_local_to_global(
                                               cd.cell_matrix,
                                               cd.local_dof_indices,
                                               mg_matrices[cd.level]);
    const unsigned dofs_per_cell = cd.local_dof_indices.size();
    
    for(unsigned i = 0; i < dofs_per_cell; ++i)
      for(unsigned j = 0; j < dofs_per_cell; ++j)
        if(mg_constrained_dofs.is_interface_matrix_entry(
              cd.level, cd.local_dof_indices[i], cd.local_dof_indices[j]))
        {
          mg_interface_matrices[cd.level].add(cd.local_dof_indices[i],
                                              cd.local_dof_indices[j],
                                              cd.cell_matrix(i, j));
        }
  };

  const unsigned n_gauss_points = fe.degree+1;

  ScratchData<dim> scratch_data(mapping,
                                fe,
                                n_gauss_points,
                                update_values | update_gradients | 
                                  update_JxW_values);

  MeshWorker::mesh_loop(dof_handler.begin_mg(),
                        dof_handler.end_mg(),
                        cell_worker,
                        copier,
                        scratch_data,
                        CopyData(),
                        MeshWorker::assemble_own_cells);
  
  mg_transfer.initialize_constraints(mg_constrained_dofs);
  mg_transfer.build(dof_handler);
}


template <int dim>
void Stokes<dim>::setup_grad_pressure()
{
  grad_pressure_u1.reinit(solution_u1.size());
  grad_pressure_u2.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();

  std::vector<Vector<double>> grad_pressure_u1_th(n_thr);
  std::vector<Vector<double>> grad_pressure_u2_th(n_thr);

#pragma omp parallel for
for(int th = 0; th < n_thr; th++)
{
  grad_pressure_u1_th[th].reinit(solution_u1.size());
  grad_pressure_u2_th[th].reinit(solution_u1.size());
  
  Vector<double> cell_rhs_u1(dofs_per_cell);
  Vector<double> cell_rhs_u2(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<Tensor<1,dim>> grad_pressure_q_point(quadrature_formula.size());
  int cnt = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cnt++;
      if(cnt % n_thr != th) continue;
      cell_rhs_u1 = 0.;
      cell_rhs_u2 = 0.;
      fe_values.reinit(cell);
      fe_values.get_function_gradients(pressure, grad_pressure_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          cell_rhs_u1(i) += weight * fe_values.shape_value(i, q_index) * 
                            grad_pressure_q_point[q_index][0];
          cell_rhs_u2(i) += weight * fe_values.shape_value(i, q_index) * 
                            grad_pressure_q_point[q_index][1];
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        grad_pressure_u1_th[th][local_dof_indices[i]] += cell_rhs_u1(i);
        grad_pressure_u2_th[th][local_dof_indices[i]] += cell_rhs_u2(i);
      }
    }
}

  for(int th = 0; th < n_thr; th++)
  {
    grad_pressure_u1 += grad_pressure_u1_th[th];
    grad_pressure_u2 += grad_pressure_u2_th[th];
  }
}


template <int dim>
void Stokes<dim>::projection_poisson_setup(const Vector<double>& u1, 
                                         const Vector<double>& u2)
{
  system_rhs.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0.;
      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          auto grad_phi = fe_values.shape_grad(i, q_index);
          cell_rhs(i) += weight * ( u1_q_point[q_index] * grad_phi[0]
                                  + u2_q_point[q_index] * grad_phi[1] );
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs[local_dof_indices[i]] += cell_rhs(i);
    }
}


template <int dim>
void Stokes<dim>::pre_projections(Vector<double>& u1, 
                                Vector<double>& u2, 
                                const int &times)
{
  Vector<double> sum_pressure(pressure.size());

  for(int T = 0; T < times; T++)
  {
    std::cout << "Pre-projections cycle " << T << std::endl;
    
    projection_poisson_setup(u1, u2);
    system_matrix_pressure.copy_from(laplace_matrix_pressure);
    constraints_pressure.condense(system_matrix_pressure, system_rhs);
    solve(pressure, true);
    sum_pressure.add(1., pressure);
    constraints_pressure.distribute(pressure);
    setup_grad_pressure();
    
    system_matrix.copy_from(mass_matrix);
    mass_matrix.vmult(system_rhs, u1);
    system_rhs.add(-1., grad_pressure_u1);
    constraints_u1.condense(system_matrix, system_rhs);
    solve(u1);
    constraints_u1.distribute(u1);
    
    system_matrix.copy_from(mass_matrix);
    mass_matrix.vmult(system_rhs, u2);
    system_rhs.add(-1., grad_pressure_u2);
    constraints_u2.condense(system_matrix, system_rhs);
    solve(u2);
    constraints_u2.distribute(u2);
    
    std::cout << std::endl;
  }
  pressure = sum_pressure;
}


template <int dim>
void Stokes<dim>::run(){
  make_mesh();
  setup_system();
  
  VectorTools::interpolate(dof_handler,
                           Initial1<dim>(),
                           solution_u1);
  VectorTools::interpolate(dof_handler,
                           Initial2<dim>(),
                           solution_u2);
  
  pre_projections(solution_u1, solution_u2, proj_steps);

  assemble_multigrid(true);
  
  mass_matrix.vmult(system_rhs, solution_u1);
  system_matrix.copy_from(laplace_matrix);
  constraints_u1.condense(system_matrix, system_rhs);
  solve(solution_u1);
  constraints_u1.distribute(solution_u1);

  mass_matrix.vmult(system_rhs, solution_u2);
  system_matrix.copy_from(laplace_matrix);
  constraints_u1.condense(system_matrix, system_rhs);
  solve(solution_u2);
  constraints_u1.distribute(solution_u2);

  next_step = false;
  output_result();
}


int main(int argc, const char *argv[]){
  omp_set_num_threads(n_thr);
  if(argc < 2)
  {
    std::cerr << "Param error!" << std::endl;
    return -1;
  }
  int proj_steps = 20;
  if(argc == 3) proj_steps = std::stoi(argv[2]);
  Stokes<2> stokes(std::stoi(argv[1]), proj_steps);
  stokes.run();
  return 0;
}
