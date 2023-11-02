//----------------------------------------------------------
// This code is for cylindrical turbulence test
// 
// Velocity-pressure decomposition method: UPPE
//   (See the formula (19) of Jianguo Liu (2010) )
// Time discretization: IMEX-trapezoidal (2nd order)
// Space discretization: Q2 element (2nd order in H1)
//----------------------------------------------------------

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

using namespace dealii;

// DO NOT Use ADAPTIVE option now!!!
// #define ADAPTIVE

// Add the following code in the viscous-box test.
#define NO_FORCING_TERM

// Remove the following code in the viscous-box test.
// #define ANALYTIC_SOLUTION

// Add the following code if you want to see how many CG iterations.
#define OUTPUT_CG_ITERATIONS

const unsigned Raynolds = 1000;
const double diffusion_coefficient = 1.0 / Raynolds;

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


//----------------------------Boundary Values----------------------------

template <int dim>
class InflowBoundaryTerm : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};

template <int dim>
double InflowBoundaryTerm<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  static const double Um = 6.0;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return Um * p[1] * (0.41-p[1]) / (0.41 * 0.41) * sin(M_PI*this->get_time()/8);
}


template <int dim>
class InflowBoundaryTermDt : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;
};

template <int dim>
double InflowBoundaryTermDt<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  (void)component;
  static const double Um = 6.0;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  return Um * p[1] * (0.41-p[1]) / (0.41 * 0.41) * cos(M_PI*this->get_time()/8) * M_PI/8;
}


//---------------------------------Solver Class-----------------------------------


template <int dim>
class INSE{
public:
  INSE(const int &, const double &);
  void run();

private:
  template <class Iterator>
  void cell_worker(const Iterator    &cell,
                   ScratchData<dim>  &scratch_data,
                   CopyData          &copy_data);
  void assemble_multigrid();

  void make_mesh();
  void refine_mesh(const unsigned& min_level, const unsigned& max_level);
  void setup_system();
  void setup_convection(const Vector<double>& u1, const Vector<double>& u2);
  void setup_grad_pressure();
  void make_constraints_u1(const double &t);
  void update_pressure(const Vector<double>& u1, const Vector<double>& u2);
  void solve_time_step(Vector<double>& solution, const bool ispressure = false);
  void compute_vortricity();
  void output_result(const bool force_output = false);

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

  Vector<double> convection_u1;
  Vector<double> convection_u2;
  Vector<double> grad_pressure_u1;
  Vector<double> grad_pressure_u2;

  Vector<double> pressure;
  Vector<double> vortricity;
  Vector<double> solution_u1;
  Vector<double> solution_u2;
  Vector<double> prev_solution_u1;
  Vector<double> prev_solution_u2;
  Vector<double> system_rhs;

  int      level;
  double   end_time;
  double   time;
  double   time_step;
  unsigned timestep_number;
};


//---------------------------------Solver Implemention-----------------------------------


template <int dim>
INSE<dim>::INSE
  (const int &N, const double &T)
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe(1)
  , dof_handler(triangulation)
  , level(N)
  , end_time(T)
  , time_step(1e-4)
{}


template <int dim>
void INSE<dim>::make_mesh(){
  // GridGenerator::hyper_cube(triangulation);

  // The following code is for cylindrical turbulence
  SphericalManifold<2> boundary(Point<2>(0.5, 0.2));
  Triangulation<2> middle, right, tmp;

  GridGenerator::subdivided_hyper_rectangle(
      right,
      std::vector<unsigned int>({9U, 2U}),
      Point<2>(0.7, 0),
      Point<2>(2.5, 0.41),
      false);

  GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
  middle.reset_all_manifolds();
  for (Triangulation<2>::cell_iterator cell = middle.begin();
       cell != middle.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
    {
      bool is_inner_rim = true;
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_face; ++v)
      {
        Point<2> &vertex = cell->face(f)->vertex(v);
        if (std::abs(vertex.distance(Point<2>(0.5, 0.2)) - 0.05) > 1e-10)
        {
          is_inner_rim = false;
          break;
        }
      }
      if (is_inner_rim)
        cell->face(f)->set_manifold_id(1);
    }
  middle.set_manifold(1, boundary);
  middle.refine_global(1);

  for (Triangulation<2>::cell_iterator cell = middle.begin();
       cell != middle.end();
       ++cell)
    for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
    {
      Point<2> &vertex = cell->vertex(v);
      if (std::abs(vertex[0] - 0.7) < 1e-10 &&
          std::abs(vertex[1] - 0.2) < 1e-10)
        vertex = Point<2>(0.7, 0.205);
      else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
               std::abs(vertex[1] - 0.3) < 1e-10)
        vertex = Point<2>(0.7, 0.41);
      else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
               std::abs(vertex[1] - 0.1) < 1e-10)
        vertex = Point<2>(0.7, 0);
      else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
               std::abs(vertex[1] - 0.4) < 1e-10)
        vertex = Point<2>(0.5, 0.41);
      else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
               std::abs(vertex[1] - 0.0) < 1e-10)
        vertex = Point<2>(0.5, 0.0);
      else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
               std::abs(vertex[1] - 0.3) < 1e-10)
        vertex = Point<2>(0.3, 0.41);
      else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
               std::abs(vertex[1] - 0.1) < 1e-10)
        vertex = Point<2>(0.3, 0);
      else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
               std::abs(vertex[1] - 0.2) < 1e-10)
        vertex = Point<2>(0.3, 0.205);
      else if (std::abs(vertex[0] - 0.57) < 0.01 &&
               std::abs(vertex[1] - 0.13) < 0.01)
        vertex = Point<2>(0.6177, 0.0823);
      else if (std::abs(vertex[0] - 0.57) < 0.01 &&
               std::abs(vertex[1] - 0.27) < 0.01)
        vertex = Point<2>(0.6177, 0.3227);
      else if (std::abs(vertex[0] - 0.43) < 0.01 &&
               std::abs(vertex[1] - 0.13) < 0.01)
        vertex = Point<2>(0.3823, 0.0823);
      else if (std::abs(vertex[0] - 0.43) < 0.01 &&
               std::abs(vertex[1] - 0.27) < 0.01)
        vertex = Point<2>(0.3823, 0.3227);
    }

  GridGenerator::flatten_triangulation(middle, tmp);
  GridGenerator::merge_triangulations(tmp, right, triangulation);

  triangulation.reset_all_manifolds();
  for (Triangulation<2>::cell_iterator cell = triangulation.begin();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
    {
      bool is_inner_rim = true;
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_face; ++v)
      {
        Point<2> &vertex = cell->face(f)->vertex(v);
        if (std::abs(vertex.distance(Point<2>(0.5, 0.2)) - 0.05) > 1e-10)
        {
          is_inner_rim = false;
          break;
        }
      }
      if (is_inner_rim)
        cell->face(f)->set_manifold_id(1);
    }
  triangulation.set_manifold(1, boundary);

  triangulation.refine_global(level);

  for (Triangulation<2>::active_cell_iterator cell = triangulation.begin();
       cell != triangulation.end();
       ++cell)
  {
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
      {
        if (std::abs(cell->face(f)->center()[0] - 2.5) < 1e-10)
          cell->face(f)->set_all_boundary_ids(4);
        else if (std::abs(cell->face(f)->center()[0] - 0.3) < 1e-10)
          cell->face(f)->set_all_boundary_ids(2);
        else if (std::abs(cell->face(f)->center()[1] - 0.41) < 1e-10)
          cell->face(f)->set_all_boundary_ids(3);
        else if (std::abs(cell->face(f)->center()[1]) < 1e-10)
          cell->face(f)->set_all_boundary_ids(3);
        else cell->face(f)->set_all_boundary_ids(1);
      }
  }

  // Refine the grid near the tail.
  // DONNOT use it until working out with the multigrid hanging nodes problem.
  //
  // for (Triangulation<2>::active_cell_iterator cell = triangulation.begin();
  //      cell != triangulation.end();
  //      ++cell)
  // {
  //   bool is_tail = true;
  //   for (unsigned int f = 0; f < GeometryInfo<2>::vertices_per_cell; ++f)
  //     if(cell->vertex(f)[0] < 2.3) is_tail = false;
  //   if(is_tail)
  //     cell->set_refine_flag();
  // }
  // triangulation.prepare_coarsening_and_refinement();
  // triangulation.execute_coarsening_and_refinement();

  std::cerr << "make_mesh done. cell: " << triangulation.n_active_cells() << std::endl;
}


template <int dim>
void INSE<dim>::refine_mesh(const unsigned &min_level, const unsigned &max_level)
{
  Vector<float> estimated_error_u1_per_cell(triangulation.n_active_cells());
  Vector<float> estimated_error_u2_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution_u1,
                                     estimated_error_u1_per_cell);
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},                                 // Here for Neumann conditions.
                                     solution_u2,
                                     estimated_error_u2_per_cell);
  // Combine the results to compute the error of u=(u1,u2)
  for(unsigned i = 0; i < estimated_error_u1_per_cell.size(); i++)
  {
    estimated_error_u1_per_cell[i] = estimated_error_u1_per_cell[i] * estimated_error_u1_per_cell[i]
                                   + estimated_error_u2_per_cell[i] * estimated_error_u2_per_cell[i];
  }
  
  GridRefinement::refine_and_coarsen_optimize(triangulation,
                                              estimated_error_u1_per_cell);
  if(triangulation.n_levels() > max_level)
    for(const auto &cell : triangulation.active_cell_iterators_on_level(max_level))
      cell->clear_refine_flag();
  for(const auto &cell : triangulation.active_cell_iterators_on_level(min_level))
    cell->clear_coarsen_flag();
  
  SolutionTransfer<dim> solution_trans_u1(dof_handler);
  SolutionTransfer<dim> solution_trans_u2(dof_handler);
  SolutionTransfer<dim> solution_trans_pressure(dof_handler);

  Vector<double> previous_solution_u1;
  Vector<double> previous_solution_u2;
  Vector<double> previous_solution_pressure;

  previous_solution_u1 = solution_u1;
  previous_solution_u2 = solution_u2;
  previous_solution_pressure = pressure;

  triangulation.prepare_coarsening_and_refinement();
  solution_trans_u1.prepare_for_coarsening_and_refinement(previous_solution_u1);
  solution_trans_u2.prepare_for_coarsening_and_refinement(previous_solution_u2);
  solution_trans_pressure.prepare_for_coarsening_and_refinement(previous_solution_pressure);
  triangulation.execute_coarsening_and_refinement();

  setup_system();

  solution_trans_u1.interpolate(previous_solution_u1, solution_u1);
  solution_trans_u2.interpolate(previous_solution_u2, solution_u2);
  solution_trans_pressure.interpolate(previous_solution_pressure, pressure);

  make_constraints_u1(time);

  constraints_u1.distribute(solution_u1);
  constraints_u2.distribute(solution_u2);
  constraints_pressure.distribute(pressure);
}


template <int dim>
void INSE<dim>::compute_vortricity()
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
  vortricity = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_val_vel.reinit(cell);
      cell->get_dof_indices(ldi);
      fe_val_vel.get_function_gradients(solution_u1, grad_u1);
      fe_val_vel.get_function_gradients(solution_u2, grad_u2);
      loc_rot = 0.;
      for (unsigned int q = 0; q < nqp; ++q)
        for (unsigned int i = 0; i < dpc; ++i)
          loc_rot(i) += (grad_u2[q][0] - grad_u1[q][1]) * //
                        fe_val_vel.shape_value(i, q) *    //
                        fe_val_vel.JxW(q);

      for (unsigned int i = 0; i < dpc; ++i)
        system_rhs(ldi[i]) += loc_rot(i);
    }

  solve_time_step(vortricity);
}


template <int dim>
void INSE<dim>::output_result(const bool force_output)
{
  if(!force_output && (timestep_number % 50)) return;
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u1, "solution_u1");
  data_out.add_data_vector(solution_u2, "solution_u2");
  data_out.add_data_vector(pressure, "pressure");
  compute_vortricity();
  data_out.add_data_vector(vortricity, "vortricity");

#ifdef ANALYTIC_SOLUTION
  Vector<double> difference_per_cell(triangulation.n_active_cells());
  AnalyticSolutionU1<dim> u1;
  u1.set_time(time);
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution_u1,
                                    u1,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  data_out.add_data_vector(difference_per_cell, "error_u1");
  std::cout << "u1 L2-norm error: " << L2_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  AnalyticSolutionU2<dim> u2;
  u2.set_time(time);
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    solution_u2,
                                    u2,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  data_out.add_data_vector(difference_per_cell, "error_u2");
  std::cout << "u2 L2-norm error: " << L2_error << std::endl;

  difference_per_cell.reinit(triangulation.n_active_cells());
  AnalyticSolutionPressure<dim> p;
  p.set_time(time);
  VectorTools::integrate_difference(MappingFE<2>(fe),
                                    dof_handler,
                                    pressure,
                                    p,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  data_out.add_data_vector(difference_per_cell, "error_p");
  std::cout << "p L2-norm error: " << L2_error << std::endl;
#endif

  data_out.build_patches();
  std::ofstream output("solution/solution-" + std::to_string(timestep_number) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void INSE<dim>::solve_time_step(Vector<double>& solution, const bool ispressure)
{
  SolverControl            solver_control(2000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  if(!ispressure)
  {
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  } else {
    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    SolverControl coarse_solver_control(5000, 1e-12, false, false);
    SolverCG<Vector<double>> coarse_solver(coarse_solver_control);
    PreconditionIdentity id;
    MGCoarseGridIterativeSolver<Vector<double>,
                                SolverCG<Vector<double>>,
                                SparseMatrix<double>,
                                PreconditionIdentity>
        coarse_grid_solver(coarse_solver, mg_matrices[0], id);

    // FullMatrix<double> coarse_matrix;
    // coarse_matrix.copy_from(mg_matrices[0]);
    // MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
    // coarse_grid_solver.initialize(coarse_matrix);

    using Smoother = PreconditionChebyshev<SparseMatrix<double>, Vector<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;
    mg_smoother.initialize(mg_matrices);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);

    Multigrid<Vector<double>> mg(
      mg_matrix, coarse_grid_solver, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG< dim, Vector<double>, MGTransferPrebuilt<Vector<double>> >
      preconditioner(dof_handler, mg, mg_transfer);
    
    solver.solve(system_matrix_pressure, solution, system_rhs, preconditioner);

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
void INSE<dim>::make_constraints_u1(const double &t)
{
  constraints_u1.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           3,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u1);
  InflowBoundaryTerm<dim> boundary;
  boundary.set_time(t);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           boundary,
                                           constraints_u1);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           4,
                                           boundary,
                                           constraints_u1);
  constraints_u1.close();
}


template <int dim>
void INSE<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << std::endl
            << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  make_constraints_u1(time);

  constraints_u2.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_u2);
  auto gg = DoFTools::extract_hanging_node_dofs(dof_handler);
  for(auto x : gg)
    std::cerr << x << " ";
  std::cerr << std::endl;
  
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           3,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_u2);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           4,
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
  laplace_matrix *= diffusion_coefficient;
  
  vortricity.reinit(dof_handler.n_dofs());
  pressure.reinit(dof_handler.n_dofs());
  solution_u1.reinit(dof_handler.n_dofs());
  solution_u2.reinit(dof_handler.n_dofs());
  prev_solution_u1.reinit(dof_handler.n_dofs());
  prev_solution_u2.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  assemble_multigrid();
}


template <int dim>
template <class Iterator>
void INSE<dim>::cell_worker(const Iterator     &cell,
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
void INSE<dim>::assemble_multigrid()
{
  dof_handler.distribute_mg_dofs();
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);
  
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
      boundary_constraints[level].close();
    }

  std::cerr << "multigrid hangings" << std::endl;
  auto vvv = mg_constrained_dofs.get_refinement_edge_indices(n_levels-1);
  for(auto x : vvv)
    std::cerr << x << " ";
  std::cerr << std::endl;
  std::cerr << "---------------" << std::endl;

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
}


template <int dim>
void INSE<dim>::setup_convection
      (const Vector<double>& u1, const Vector<double>& u2)
{
  convection_u1.reinit(solution_u1.size());
  convection_u2.reinit(solution_u1.size());

#ifdef NO_CONVECTION
  return;
#endif

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_convection_u1(dofs_per_cell);
  Vector<double> cell_convection_u2(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u1_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u2_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_convection_u1 = 0.;
      cell_convection_u2 = 0.;

      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);
      fe_values.get_function_gradients(u1, grad_u1_q_point);
      fe_values.get_function_gradients(u2, grad_u2_q_point);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          cell_convection_u1(i) += weight * fe_values.shape_value(i, q_index) * 
                                  ( grad_u1_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u1_q_point[q_index][1] * u2_q_point[q_index]);
          cell_convection_u2(i) += weight * fe_values.shape_value(i, q_index) * 
                                  ( grad_u2_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u2_q_point[q_index][1] * u2_q_point[q_index]);
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
      {
        convection_u1[local_dof_indices[i]] += cell_convection_u1[i];
        convection_u2[local_dof_indices[i]] += cell_convection_u2[i];
      }
    }
}


template <int dim>
void INSE<dim>::setup_grad_pressure()
{
  grad_pressure_u1.reinit(solution_u1.size());
  grad_pressure_u2.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_rhs_u1(dofs_per_cell);
  Vector<double> cell_rhs_u2(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values);
  std::vector<Tensor<1,dim>> grad_pressure_q_point(quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
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
        grad_pressure_u1[local_dof_indices[i]] += cell_rhs_u1(i);
        grad_pressure_u2[local_dof_indices[i]] += cell_rhs_u2(i);
      }
    }
}


template <int dim>
void INSE<dim>::update_pressure(
  const Vector<double> &u1, const Vector<double>& u2)
{
  system_matrix_pressure.copy_from(laplace_matrix_pressure);
  system_rhs.reinit(solution_u1.size());

  const unsigned dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  QGauss<dim> quadrature_formula(fe.degree+1);
  QGauss<dim-1> quadrature_formula_face(fe.degree+1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);
  const MappingQ<dim> mapping(fe.degree);
  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
                                   quadrature_formula_face,
                                   update_values | update_gradients |
                                   update_JxW_values | update_normal_vectors
                                    | update_quadrature_points);

  std::vector<double> u1_q_point(quadrature_formula.size());
  std::vector<double> u2_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u1_q_point(quadrature_formula.size());
  std::vector<Tensor<1,dim>> grad_u2_q_point(quadrature_formula.size());

  std::vector<Tensor<1,dim>> grad_u1_face_q_point(quadrature_formula_face.size());
  std::vector<Tensor<1,dim>> grad_u2_face_q_point(quadrature_formula_face.size());

#ifndef NO_FORCING_TERM
  ForcingTerm1<dim> forcing_u1_t1;
  ForcingTerm2<dim> forcing_u2_t1;
  forcing_u1_t1.set_time(time);
  forcing_u2_t1.set_time(time);
#endif

  InflowBoundaryTermDt<dim> boundary_dt;
  boundary_dt.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0.;
      fe_values.reinit(cell);
      fe_values.get_function_values(u1, u1_q_point);
      fe_values.get_function_values(u2, u2_q_point);
      fe_values.get_function_gradients(u1, grad_u1_q_point);
      fe_values.get_function_gradients(u2, grad_u2_q_point);
      auto q_points = fe_values.get_quadrature_points();

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        double weight = fe_values.JxW(q_index);
        for (const unsigned int i : fe_values.dof_indices())
        {
          auto grad = fe_values.shape_grad(i, q_index);
          cell_rhs(i)    -= weight * grad[0] * 
                                  ( grad_u1_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u1_q_point[q_index][1] * u2_q_point[q_index])
                          + weight * grad[1] * 
                                  ( grad_u2_q_point[q_index][0] * u1_q_point[q_index] + 
                                    grad_u2_q_point[q_index][1] * u2_q_point[q_index]);

#ifndef NO_FORCING_TERM
          double force_u1 = forcing_u1_t1.value(q_points[q_index]);
          double force_u2 = forcing_u2_t1.value(q_points[q_index]);
          cell_rhs(i)  +=  weight * grad[0] * force_u1
                        +  weight * grad[1] * force_u2;
#endif
        }
      }

      for (const auto &f : cell->face_iterators()){
        if( ! (f->at_boundary()) ) continue;
        fe_face_values.reinit(cell, f);
        fe_face_values.get_function_gradients(u1, grad_u1_face_q_point);
        fe_face_values.get_function_gradients(u2, grad_u2_face_q_point);
        auto q_face_points = fe_face_values.get_quadrature_points();
        
        for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
        {
          auto normal = fe_face_values.normal_vector(q_index);
          // if(fabs(normal[0])>1e-6 && fabs(normal[1])>1e-6)
          //   std::cout << q_face_points[q_index] << " " << normal << std::endl;
          double weight = fe_face_values.JxW(q_index);

          auto gdt = boundary_dt.value(q_face_points[q_index]);
          if(f->boundary_id() != 2 && f->boundary_id() != 4) gdt = 0;

          for (const unsigned int i : fe_face_values.dof_indices())
          {
            auto grad = fe_face_values.shape_grad(i, q_index);
            double normal_grad_phi = normal[0]*grad[1] - normal[1]*grad[0];
            double vor_u = grad_u2_face_q_point[q_index][0] - grad_u1_face_q_point[q_index][1];
            cell_rhs(i)    += diffusion_coefficient * vor_u * normal_grad_phi * weight
                            - normal[0] * gdt * fe_face_values.shape_value(i,q_index) * weight;
          }
        }
      }

      // distribute to global
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs[local_dof_indices[i]] += cell_rhs(i);
    }
  
  constraints_pressure.condense(system_matrix_pressure, system_rhs);
  solve_time_step(pressure, true);
  constraints_pressure.distribute(pressure);
}


template <int dim>
void INSE<dim>::run(){

#ifdef ADAPTIVE
  const unsigned initial_global_refinement = 1 + level;
  const unsigned n_adaptive_pre_refinement_steps = 4;
#else
  const unsigned initial_global_refinement = 0;
#endif

  Vector<double> tmp;
  Vector<double> middle_solution_u1;
  Vector<double> middle_solution_u2;
  Vector<double> forcing_term_1;
  Vector<double> forcing_term_2;

  Vector<double> grad_pressure_u1_stage1;
  Vector<double> grad_pressure_u2_stage1;
  Vector<double> convection_u1_stage1;
  Vector<double> convection_u2_stage1;

#ifdef ADAPTIVE
  unsigned pre_refinement_step = 0;
#endif

  make_mesh();
  triangulation.refine_global(initial_global_refinement);
  setup_system();

#ifdef ADAPTIVE
start_time_iteration:
#endif

  time            = 0.0;
  timestep_number = 0;

  tmp.reinit(solution_u1.size());
  middle_solution_u1.reinit(solution_u1.size());
  middle_solution_u2.reinit(solution_u1.size());
  forcing_term_1.reinit(solution_u1.size());
  forcing_term_2.reinit(solution_u1.size());

#ifndef NO_FORCING_TERM
  ForcingTerm1<dim> rhs_func_1;
  ForcingTerm2<dim> rhs_func_2;
#endif

  Functions::ZeroFunction<dim> initial_1;
  // InflowBoundaryTerm<dim> initial_1;
  Functions::ZeroFunction<dim> initial_2;
  
  VectorTools::interpolate(dof_handler,
                           initial_1,
                           prev_solution_u1);
  VectorTools::interpolate(dof_handler,
                           initial_2,
                           prev_solution_u2);
  solution_u1 = prev_solution_u1;
  solution_u2 = prev_solution_u2;

  while(time <= end_time){
    // ------------------------------first stage-------------------------------------
    setup_convection(prev_solution_u1, prev_solution_u2);
    update_pressure(prev_solution_u1, prev_solution_u2);
    output_result();

    time += time_step;
    timestep_number++;
    make_constraints_u1(time);
#ifdef OUTPUT_CG_ITERATIONS
    std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
#else
    if(timestep_number % 50 == 0)
      std::cout << std::endl << "Time step " << timestep_number << " at t=" << time << std::endl;
#endif

    setup_grad_pressure();

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, prev_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u1);
    system_rhs.add(-time_step, grad_pressure_u1);

#ifndef NO_FORCING_TERM
    rhs_func_1.set_time(time-time_step);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_1,
                                        forcing_term_1);
    system_rhs.add(time_step, forcing_term_1);
#endif

    constraints_u1.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u1);
    constraints_u1.distribute(middle_solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(0.5*time_step, laplace_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, prev_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-time_step, convection_u2);
    system_rhs.add(-time_step, grad_pressure_u2);

#ifndef NO_FORCING_TERM
    rhs_func_2.set_time(time-time_step);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_2,
                                        forcing_term_2);
    system_rhs.add(time_step, forcing_term_2);
#endif
    
    constraints_u2.condense(system_matrix, system_rhs);
    solve_time_step(middle_solution_u2);
    constraints_u2.distribute(middle_solution_u2);


    //------------------------------second stage-------------------------------------

    convection_u1_stage1 = convection_u1;
    convection_u2_stage1 = convection_u2;
    grad_pressure_u1_stage1 = grad_pressure_u1;
    grad_pressure_u2_stage1 = grad_pressure_u2;

    setup_convection(middle_solution_u1, middle_solution_u2);
    update_pressure(middle_solution_u1, middle_solution_u2);
    setup_grad_pressure();

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u1);
    laplace_matrix.vmult(tmp, prev_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, middle_solution_u1);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-0.5*time_step, convection_u1);
    system_rhs.add(-0.5*time_step, convection_u1_stage1);
    system_rhs.add(-0.5*time_step, grad_pressure_u1);
    system_rhs.add(-0.5*time_step, grad_pressure_u1_stage1);

#ifndef NO_FORCING_TERM
    rhs_func_1.set_time(time);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_1,
                                        tmp);
    system_rhs.add(0.5*time_step, forcing_term_1);
    system_rhs.add(0.5*time_step, tmp);
#endif

    constraints_u1.condense(system_matrix, system_rhs);
    solve_time_step(solution_u1);
    constraints_u1.distribute(solution_u1);

    // setup system_matrix
    system_matrix.copy_from(mass_matrix);

    // setup right side term
    mass_matrix.vmult(system_rhs, prev_solution_u2);
    laplace_matrix.vmult(tmp, middle_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    laplace_matrix.vmult(tmp, prev_solution_u2);
    system_rhs.add(-0.5*time_step, tmp);
    system_rhs.add(-0.5*time_step, convection_u2);
    system_rhs.add(-0.5*time_step, convection_u2_stage1);
    system_rhs.add(-0.5*time_step, grad_pressure_u2);
    system_rhs.add(-0.5*time_step, grad_pressure_u2_stage1);

#ifndef NO_FORCING_TERM
    rhs_func_2.set_time(time);
    VectorTools::create_right_hand_side(dof_handler,
                                        QGauss<dim>(fe.degree+1),
                                        rhs_func_2,
                                        tmp);
    system_rhs.add(0.5*time_step, forcing_term_2);
    system_rhs.add(0.5*time_step, tmp);
#endif

    constraints_u2.condense(system_matrix, system_rhs);
    solve_time_step(solution_u2);
    constraints_u2.distribute(solution_u2);

#ifdef ADAPTIVE
    // Refine mesh 1 times per 5 time-step. At the begining, refine 5 times.
    if ((timestep_number == 1) &&
        (pre_refinement_step < n_adaptive_pre_refinement_steps))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        ++pre_refinement_step;
        std::cout << std::endl;
        std::cout << "pre refinement step: " << pre_refinement_step << std::endl;
        std::cout << std::endl;
        goto start_time_iteration;
      }
    else if ((timestep_number > 0) && (timestep_number % 5 == 0))
      {
        refine_mesh(initial_global_refinement,
                    initial_global_refinement +
                      n_adaptive_pre_refinement_steps);
        tmp.reinit(solution_u1.size());
        middle_solution_u1.reinit(solution_u1.size());
        middle_solution_u2.reinit(solution_u1.size());
        convection_u1_stage1.reinit(solution_u1.size());
        convection_u2_stage1.reinit(solution_u1.size());
        grad_pressure_u1_stage1.reinit(solution_u1.size());
        grad_pressure_u2_stage1.reinit(solution_u1.size());

#ifndef NO_FORCING_TERM
        forcing_term_1.reinit(solution_u1.size());
        forcing_term_2.reinit(solution_u1.size());
#endif
      }
#endif

    prev_solution_u1 = solution_u1;
    prev_solution_u2 = solution_u2;
  }

  setup_convection(prev_solution_u1, prev_solution_u2);
  update_pressure(prev_solution_u1, prev_solution_u2);
  std::cout << std::endl;
  output_result(true);
}


int main(int argc, const char *argv[]){
  if(argc != 3){
    std::cerr << "Param error! Please run with command" << std::endl;
    std::cerr << "./convection-diffusion N T" << std::endl;
    std::cerr << "where N is the level of base grid, T is end_time." << std::endl;
    return -1;
  }
  int level = std::stoi(argv[1]);
  double end_time = std::stod(argv[2]);
  INSE<2> inse(level, end_time);
  inse.run();
  return 0;
}