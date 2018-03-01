#include <ct/optcon/optcon.h>  // also includes ct_core
#include <my_ct_project/configDir.h>
// #include <ct/core/core.h>
#include "my_ct_project/CustomController.h"
#include "my_ct_project/plotResultsOscillator.h"

#include <ct/models/HyQ/HyQ.h>
#include <ct/models/HyA/HyA.h>
#include <ct/rbd/systems/FixBaseFDSystem.h>
#include <ct/rbd/systems/FloatingBaseFDSystem.h>
#include <ct/models/InvertedPendulum/InvertedPendulum.h>

#include <ct/rbd/rbd.h>
#include <ct/models/CodegenOutputDirs.h>

#include <memory>
#include <array>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

/*HyQ Linearisation*/
#include <ct/models/HyQ/codegen/HyQWithContactModelLinearizedForward.h>
#include <ct/models/HyQ/codegen/HyQBareModelLinearizedForward.h>


using namespace ct::core;
using namespace ct::optcon;

int main(int argc, char** argv)
{
    typedef ct::rbd::FloatingBaseFDSystem<ct::rbd::HyQ::Dynamics> HyQSystem;

    const size_t STATE_DIM = HyQSystem::STATE_DIM;
    const size_t CONTROL_DIM = HyQSystem::CONTROL_DIM;
    

    std::cout << "STATE_DIM: " << STATE_DIM << std::endl;
    std::cout << "CONTROL_DIM: " << CONTROL_DIM << std::endl;

    /* STEP 1: set up the Nonlinear Optimal Control Problem
	 * First of all, we need to create instances of the system dynamics, the linearized system and the cost function. */

    /* STEP 1-A: create a instance of the oscillator dynamics for the optimal control problem.
	 * Please also compare the documentation of SecondOrderSystem.h */
    std::shared_ptr<::ControlledSystem<STATE_DIM, CONTROL_DIM>> HyQDynamics(
        new ct::rbd::FloatingBaseFDSystem<ct::rbd::HyQ::Dynamics>);


    /* STEP 1-B: Although the first order derivatives of the oscillator are easy to derive, let's illustrate the use of the System Linearizer,
	 * which performs numerical differentiation by the finite-difference method. The system linearizer simply takes the
	 * the system dynamics as argument. Alternatively, you could implement your own first-order derivatives by overloading the class LinearSystem.h */
    std::shared_ptr<::SystemLinearizer<STATE_DIM, CONTROL_DIM>> adLinearizer(
        new ::SystemLinearizer<STATE_DIM, CONTROL_DIM>(HyQDynamics));

    /* STEP 1-C: create a cost function. We have pre-specified the cost-function weights for this problem in "nlocCost.info".
	 * Here, we show how to create terms for intermediate and final cost and how to automatically load them from the configuration file.
	 * The verbose option allows to print information about the loaded terms on the terminal. */
    std::shared_ptr<::TermQuadratic<STATE_DIM, CONTROL_DIM>> intermediateCost(
        new ::TermQuadratic<STATE_DIM, CONTROL_DIM>());
    std::shared_ptr<::TermQuadratic<STATE_DIM, CONTROL_DIM>> finalCost(
        new ::TermQuadratic<STATE_DIM, CONTROL_DIM>());
    bool verbose = false;
    intermediateCost->loadConfigFile(::configDir + "/nlocCost.info", "intermediateCost", verbose);
    finalCost->loadConfigFile(::configDir + "/nlocCost.info", "finalCost", verbose);

    // Since we are using quadratic cost function terms in this example, the first and second order derivatives are immediately known and we
    // define the cost function to be an "Analytical Cost Function". Let's create the corresponding object and add the previously loaded
    // intermediate and final term.
    std::shared_ptr<::CostFunctionQuadratic<STATE_DIM, CONTROL_DIM>> costFunction(
        new ::CostFunctionAnalytical<STATE_DIM, CONTROL_DIM>());
    costFunction->addIntermediateTerm(intermediateCost);
    costFunction->addFinalTerm(finalCost);


    /* STEP 1-D: initialization with initial state and desired time horizon */

    ::StateVector<STATE_DIM> x0;
    x0.setRandom();  // in this example, we choose a random initial state x0

    ::Time timeHorizon = 3.0;  // and a final time horizon in [sec]

    // STEP 1-E: create and initialize an "optimal control problem"
    ::OptConProblem<STATE_DIM, CONTROL_DIM> optConProblem(
        timeHorizon, x0, HyQDynamics, costFunction, adLinearizer);


    /* STEP 2: set up a nonlinear optimal control solver. */

    /* STEP 2-A: Create the settings.
	 * the type of solver, and most parameters, like number of shooting intervals, etc.,
	 * can be chosen using the following settings struct. Let's use, the iterative
	 * linear quadratic regulator, iLQR, for this example. In the following, we
	 * modify only a few settings, for more detail, check out the NLOptConSettings class. */
    ::NLOptConSettings ilqr_settings;
    ilqr_settings.dt = 0.01;  // the control discretization in [sec]
    ilqr_settings.integrator = ::IntegrationType::EULERCT;
    ilqr_settings.discretization = ::NLOptConSettings::APPROXIMATION::FORWARD_EULER;
    ilqr_settings.max_iterations = 1;
    ilqr_settings.nThreads = 1;  // use multi-threading
    ilqr_settings.nlocp_algorithm = ::NLOptConSettings::NLOCP_ALGORITHM::ILQR;
    ilqr_settings.lqocp_solver =
        ::NLOptConSettings::LQOCP_SOLVER::GNRICCATI_SOLVER;  // solve LQ-problems using custom Riccati solver
    ilqr_settings.printSummary = true;
    ilqr_settings.debugPrint = true;

    std::cout << "AAAAAAAAA" << std::endl;

    /* STEP 2-B: provide an initial guess */
    // calculate the number of time steps K
    size_t K = ilqr_settings.computeK(timeHorizon);


    /* design trivial initial controller for iLQR. Note that in this simple example,
	 * we can simply use zero feedforward with zero feedback gains around the initial position.
	 * In more complex examples, a more elaborate initial guess may be required.*/
    ::FeedbackArray<STATE_DIM, CONTROL_DIM> u0_fb(K, ::FeedbackMatrix<STATE_DIM, CONTROL_DIM>::Zero());
    ::ControlVectorArray<CONTROL_DIM> u0_ff(K, ::ControlVector<CONTROL_DIM>::Zero());
    ::StateVectorArray<STATE_DIM> x_ref_init(K + 1, x0);
    ::NLOptConSolver<STATE_DIM, CONTROL_DIM>::Policy_t initController(x_ref_init, u0_ff, u0_fb, ilqr_settings.dt);

    std::cout << "BBBBBBBBBBB" << std::endl;

    // STEP 2-C: create an NLOptConSolver instance
    ::NLOptConSolver<STATE_DIM, CONTROL_DIM> iLQR(optConProblem, ilqr_settings);

    std::cout << "CCCCCCCCCC" << std::endl;

    // set the initial guess
    iLQR.setInitialGuess(initController);

    std::cout << "DDDDDDDDDD" << std::endl;

    // STEP 3: solve the optimal control problem
    iLQR.solve();



    // STEP 4: retrieve the solution
    ::StateFeedbackController<STATE_DIM, CONTROL_DIM> solution = iLQR.getSolution();

    // let's plot the output
    plotResultsOscillator<STATE_DIM, CONTROL_DIM>(solution.x_ref(), solution.uff(), solution.time());

    return 1;

}
