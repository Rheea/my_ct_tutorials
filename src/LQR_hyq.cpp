#include <ct/optcon/optcon.h>  // also includes ct_core
#include <my_ct_project/configDir.h>
#include <ct/core/core.h>
#include "my_ct_project/CustomController.h"
#include <ct/models/HyQ/HyQ.h>
#include <ct/models/HyA/HyA.h>
#include <ct/rbd/systems/FixBaseFDSystem.h>
#include <ct/rbd/systems/FloatingBaseFDSystem.h>
#include <ct/models/InvertedPendulum/InvertedPendulum.h>

#include <ct/optcon/optcon.h>
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


using namespace ct;
using namespace ct::rbd;
using namespace ct::models::HyQ;


int main(int argc, char** argv)
{
    typedef FloatingBaseFDSystem<HyQ::Dynamics> HyQSystem;

    const size_t STATE_DIM = HyQSystem::STATE_DIM;
    const size_t CONTROL_DIM = HyQSystem::CONTROL_DIM;
    
    std::shared_ptr<HyQSystem> hyqSystem(new HyQSystem);
    std::shared_ptr<HyQSystem> hyqSystem2(new HyQSystem);

    RbdLinearizer<HyQSystem> rbdLinearizer(hyqSystem, true);
    core::SystemLinearizer<STATE_DIM, CONTROL_DIM> systemLinearizer(hyqSystem2, true);
    ct::models::HyQ::HyQWithContactModelLinearizedForward hyqLinear;
    // ct::models::HyQ::HyQBareModelLinearizedForward hyqLinear_tmp;

    std::cout << "STATE_DIM: " << STATE_DIM << std::endl;
    std::cout << "CONTROL_DIM: " << CONTROL_DIM << std::endl;


    // // create an auto-differentiable instance of the oscillator dynamics
    // ct::core::ADCGScalar w_n(50.0);
    // std::shared_ptr<ct::core::ControlledSystem<state_dim, control_dim, ct::core::ADCGScalar>> oscillatorDynamics(
    //     new ct::core::tpl::SecondOrderSystem<ct::core::ADCGScalar>(w_n));

    // // create an Auto-Differentiation Linearizer with code generation on the quadrotor model
    // ct::core::ADCodegenLinearizer<state_dim, control_dim> adLinearizer(oscillatorDynamics);

    // compile the linearized model just-in-time
    // hyqLinear.compileJIT();

    // define the linearization point around steady state
    ct::core::StateVector<STATE_DIM> x;
    x.setZero();
    ct::core::ControlVector<CONTROL_DIM> u;
    u.setZero();
    double t = 0.0;

    // compute the linearization around the nominal state using the Auto-Diff Linearizer
    auto A = hyqLinear.getDerivativeState(x, u, t);
    auto B = hyqLinear.getDerivativeControl(x, u, t);

    // load the weighting matrices
    ct::optcon::TermQuadratic<STATE_DIM, CONTROL_DIM> quadraticCost;
    quadraticCost.loadConfigFile(ct::optcon::configDir + "/lqrCost.info", "termLQR");
    auto Q = quadraticCost.stateSecondDerivative(x, u, t);    // x, u and t can be arbitrary here
    auto R = quadraticCost.controlSecondDerivative(x, u, t);  // x, u and t can be arbitrary here

    // design the LQR controller
    ct::optcon::LQR<STATE_DIM, CONTROL_DIM> lqrSolver;
    ct::core::FeedbackMatrix<STATE_DIM, CONTROL_DIM> K;

    std::cout << "A: " << std::endl << A << std::endl << std::endl;
    std::cout << "B: " << std::endl << B << std::endl << std::endl;
    std::cout << "Q: " << std::endl << Q << std::endl << std::endl;
    std::cout << "R: " << std::endl << R << std::endl << std::endl;

    lqrSolver.compute(Q, R, A, B, K);

    std::cout << "LQR gain matrix:" << std::endl << K << std::endl;

    return 1;
}
