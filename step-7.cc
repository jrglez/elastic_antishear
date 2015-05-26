/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth and Ralf Hartmann, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>


#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>

namespace Step7
{
  using namespace dealii;

  template <int dim>
  class Neumann_BC : public Function<dim>
  {
  public:
    Neumann_BC () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void value_list (const std::vector<Point<dim> >      &points,
                             std::vector<double>                 &values,
                             const unsigned int                  component = 0) const;
  };


  template <int dim>
  double Neumann_BC<dim>::value (const Point<dim>   &p,
                                    const unsigned int) const
  {
//    if (std::fabs(p[0]-1.0) < 1e-6)
//      return -1.0;
//    else
//      return 1.0;
    return 1.0;
  }

  template <int dim>
  void Neumann_BC<dim>::value_list (const std::vector<Point<dim> >   &points,
                               std::vector<double>              &values,
                               const unsigned int               component) const
  {
    const unsigned int n_points = points.size();
    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));
    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
      values[i] = Neumann_BC::value(points[i]);
  }


    template <int dim>
    class Dirichlet_BC : public Function<dim>
    {
    public:
        Dirichlet_BC () : Function<dim>() {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
      virtual void value_list (const std::vector<Point<dim> >      &points,
                               std::vector<double>                 &values,
                               const unsigned int                  component = 0) const;
    };


    template <int dim>
    double Dirichlet_BC<dim>::value (const Point<dim>   &p,
                                      const unsigned int) const
    {
  //    if (std::fabs(p[0]-1.0) < 1e-6)
  //      return -1.0;
  //    else
  //      return 1.0;
      return 1.0;
    }

    template <int dim>
    void Dirichlet_BC<dim>::value_list (const std::vector<Point<dim> >   &points,
                                 std::vector<double>              &values,
                                 const unsigned int               component) const
    {
      const unsigned int n_points = points.size();
      Assert (values.size() == n_points,
              ExcDimensionMismatch (values.size(), n_points));
      Assert (component == 0,
              ExcIndexRange (component, 0, 1));

      for (unsigned int i=0; i<n_points; ++i)
        values[i] = Dirichlet_BC::value(points[i]);
    }



  template <int dim>
  class HelmholtzProblem
  {
  public:
    enum RefinementMode
    {
      global_refinement, adaptive_refinement
    };

    HelmholtzProblem (const FiniteElement<dim> &fe,
                      const RefinementMode      refinement_mode,
                      const double width,
                      const double height,
                      const double fault_fraction);

    ~HelmholtzProblem ();

    void run ();

  private:
    void make_grid ();
    void output_initial_mesh ();
    void refine_grid ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results ();



    Triangulation<dim>                      triangulation;
    DoFHandler<dim>                         dof_handler;

    SmartPointer<const FiniteElement<dim> > fe;

    ConstraintMatrix                        constraints;

    SparsityPattern                         sparsity_pattern;
    SparseMatrix<double>                    system_matrix;

    Vector<double>                          solution;
    Vector<double>                          system_rhs;

    std::string                             filename;

    const RefinementMode                    refinement_mode;

    const double                            width;
    const double                            height;
    const double                            fault_fraction;
  };




  template <int dim>
  HelmholtzProblem<dim>::HelmholtzProblem (const FiniteElement<dim> &fe,
                                           const RefinementMode refinement_mode,
                                           const double width,
                                           const double height,
                                           const double fault_fraction) :
    dof_handler (triangulation),
    fe (&fe),
    refinement_mode (refinement_mode),
    width(width),
    height(height),
    fault_fraction(fault_fraction)
  {}



  template <int dim>
  HelmholtzProblem<dim>::~HelmholtzProblem ()
  {
    dof_handler.clear ();
  }


  template <int dim>
  void HelmholtzProblem<dim>::make_grid ()
  {
    Assert (dim == 2, ExcNotImplemented ());

    /* The domain is a horizontal rectangle, but we want the cells go be as square as possible.*/
    Point<dim> corner_low_left(0.0,-height);
    Point<dim> corner_up_right(width,0);

    std::vector<unsigned int> repetitions(dim,1);
    repetitions[0] = std::max(1.0,round(width/height));
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               corner_low_left,
                                               corner_up_right);
    triangulation.refine_global (7);

    typename Triangulation<dim>::cell_iterator cell = triangulation.begin (),
        endc = triangulation.end();
    for (; cell!=endc; ++cell)
      for (unsigned int face_number=0;
           face_number<GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        /* There are four different boundary domains: The locked part above the fault, the fault, top and
         * bottom boundaries, and the left boundary.
         * There are 4 different boundary conditions: 0) homogeneous Dirichlet (locked), 1) non-homogeneous
         * Dirichlet (imposed displacement), 2) Homogeneous Neumann (free surface), and 3) non-homogeneous
         * Neumann (imposed traction).
         */
        if ((std::fabs(cell->face(face_number)->center()(0) - (0)) < 1e-12)
            &&
            (cell->face(face_number)->center()(1) > -height*fault_fraction))
          cell->face(face_number)->set_boundary_indicator (0);
        else if ((std::fabs(cell->face(face_number)->center()(0) - (0)) < 1e-12)
            &&
            (cell->face(face_number)->center()(1) <= -height*fault_fraction))
          cell->face(face_number)->set_boundary_indicator (1);
        else if ((std::fabs(cell->face(face_number)->center()(1) - (0)) < 1e-12)
            ||
            (std::fabs(cell->face(face_number)->center()(1) - (-height)) < 1e-12))
          cell->face(face_number)->set_boundary_indicator (2);
        else if (std::fabs(cell->face(face_number)->center()(0) - (width)) < 1e-12)
          cell->face(face_number)->set_boundary_indicator (2);
  //      else if ((std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12))
  //        cell->face(face_number)->set_boundary_indicator (2);

  }


  template <int dim>
  void HelmholtzProblem<dim>::output_initial_mesh ()
  {
    std::string eps_filename;
    switch (refinement_mode)
      {
      case global_refinement:
        filename = "solution-global";
        break;
      case adaptive_refinement:
        filename = "solution-adaptive";
        break;
      default:
        Assert (false, ExcNotImplemented());
      }

    switch (fe->degree)
      {
      case 1:
        filename += "-q1";
        break;
      case 2:
        filename += "-q2";
        break;

      default:
        Assert (false, ExcNotImplemented());
      }

    eps_filename = filename + ".eps";
    std::ofstream output (eps_filename.c_str());

    GridOut grid_out;
    grid_out.write_eps (triangulation, output);
  }


  template <int dim>
  void HelmholtzProblem<dim>::refine_grid ()
  {
    switch (refinement_mode)
      {
      case global_refinement:
      {
        triangulation.refine_global (1);
        break;
      }

      case adaptive_refinement:
      {
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(3),
                                            typename FunctionMap<dim>::type(),
                                            solution,
                                            estimated_error_per_cell);

        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         estimated_error_per_cell,
                                                         0.3, 0.03);

        triangulation.execute_coarsening_and_refinement ();

        break;
      }

      default:
      {
        Assert (false, ExcNotImplemented());
      }
      }
  }


  template <int dim>
  void HelmholtzProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (*fe);
    DoFRenumbering::Cuthill_McKee (dof_handler);

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  Dirichlet_BC<dim>(),
                                                  constraints);
    constraints.close ();

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(c_sparsity);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }



  template <int dim>
  void HelmholtzProblem<dim>::assemble_system ()
  {
    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);

    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    FEValues<dim>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);

    const Neumann_BC<dim> neumann_bc;
    std::vector<double>  nbc_values (n_face_q_points);


    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);



        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                                     fe_values.shape_grad(j,q_point)*
                                     fe_values.JxW(q_point));
            }

        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)

          if (cell->face(face_number)->at_boundary()
              &&
              (cell->face(face_number)->boundary_indicator() == 3))
            {
              // Apply homogeneous Neumann boundary conditions
              fe_face_values.reinit (cell, face_number);
              neumann_bc.value_list (fe_face_values.get_quadrature_points(), nbc_values);

              for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    cell_rhs(i) += (nbc_values[q_point] *
                                    fe_face_values.shape_value(i,q_point) *
                                    fe_face_values.JxW(q_point));
            }



        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
      }
  }



  template <int dim>
  void HelmholtzProblem<dim>::solve ()
  {
    SolverControl           solver_control (4000, 1e-12);
    SolverCG<>              solver (solver_control);

    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());

    constraints.distribute (solution);
  }


  template <int dim>
  void HelmholtzProblem<dim>::output_results ()
  {
    std::string vtk_filename;

    vtk_filename = filename + ".vtk";
    std::ofstream output (vtk_filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");

    data_out.build_patches (fe->degree);
    data_out.write_vtk (output);
  }


  template <int dim>
  void HelmholtzProblem<dim>::run ()
  {
    const unsigned int n_cycles = (refinement_mode==global_refinement)?9:9;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        if (cycle == 0)
          {
            make_grid ();
            output_initial_mesh ();
          }
        else
          refine_grid ();

        setup_system ();
        assemble_system ();
        solve ();

      }
    output_results ();
  }

}


int main ()
{
  const unsigned int dim = 2;

  try
    {
      using namespace dealii;
      using namespace Step7;

      deallog.depth_console (0);

      {
        std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl
                  << "=============================================" << std::endl
                  << std::endl;

        FE_Q<dim> fe(1);
        HelmholtzProblem<dim>
        helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement,4,4,0.0125);

        helmholtz_problem_2d.run ();

        std::cout << std::endl;
      }

//      {
//        std::cout << "Solving with Q1 elements, global refinement" << std::endl
//                  << "===========================================" << std::endl
//                  << std::endl;
//
//        FE_Q<dim> fe(1);
//        HelmholtzProblem<dim>
//        helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement,2.0,2.0,0.5);
//
//        helmholtz_problem_2d.run ();
//
//        std::cout << std::endl;
//      }
//
//      {
//        std::cout << "Solving with Q2 elements, global refinement" << std::endl
//                  << "===========================================" << std::endl
//                  << std::endl;
//
//        FE_Q<dim> fe(2);
//        HelmholtzProblem<dim>
//        helmholtz_problem_2d (fe, HelmholtzProblem<dim>::global_refinement,2.0,2.0,0.5);
//
//        helmholtz_problem_2d.run ();
//
//        std::cout << std::endl;
//      }
//      {
//        std::cout << "Solving with Q2 elements, adaptive refinement" << std::endl
//                  << "===========================================" << std::endl
//                  << std::endl;
//
//        FE_Q<dim> fe(2);
//        HelmholtzProblem<dim>
//        helmholtz_problem_2d (fe, HelmholtzProblem<dim>::adaptive_refinement,2.0,2.0,0.5);
//
//        helmholtz_problem_2d.run ();
//
//        std::cout << std::endl;
//      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
