#include <ct/core/core.h>
#include "my_ct_project/CustomController.h"
#include <ct/models/HyQ/HyQ.h>
#include <ct/models/HyA/HyA.h>
#include <ct/rbd/systems/FixBaseFDSystem.h>
#include <ct/rbd/systems/FloatingBaseFDSystem.h>
#include <ct/models/InvertedPendulum/InvertedPendulum.h>

#include <ct/models/HyA/codegen/HyALinearizedForward.h>
#include <ct/models/HyA/codegen/HyALinearizedReverse.h>

#include <ct/models/HyA/codegen/HyAInverseDynJacForward.h>
#include <ct/models/HyA/codegen/HyAInverseDynJacReverse.h>

#include "my_ct_project/HyAUrdfNames.h"
#include "my_ct_project/HyAJointLimits.h"

#include <ct/optcon/optcon.h>
#include <ct/rbd/rbd.h>
#include <ct/models/CodegenOutputDirs.h>

// #include "HyQ/codegen/helperFunctions.h"

#include <memory>
#include <array>

#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace ct;
using namespace ct::rbd;
// using namespace ct::models::HyA;


/*HyA Linearisation*/

int main(int argc, char** argv)

{
	// obtain the state dimension
	const size_t NSTATE = ct::rbd::HyA::Dynamics::NSTATE;
	std::cout << "NSTATE: " << NSTATE << std::endl;

	// create an instance of the system
	std::shared_ptr<ct::core::System<NSTATE>> dynamics(new ct::rbd::FixBaseFDSystem<ct::rbd::HyA::Dynamics>);
	ct::core::Integrator<NSTATE> integrator(dynamics, ct::core::IntegrationType::RK4);

	ct ::core::StateVector<NSTATE> state;
	state.setZero();
	
	// simulate 1000 steps
    double dt = 0.001;
    ct::core::Time t0 = 0.0;
    size_t nSteps = 1000;
    integrator.integrate_n_steps(state, t0, nSteps, dt);


    // print the new state
    std::cout << "HyA state after integration: " << state.transpose() << std::endl;

    typedef FixBaseFDSystem<HyA::Dynamics> HyASystem;

    const size_t STATE_DIM = HyASystem::STATE_DIM;
    const size_t CONTROL_DIM = HyASystem::CONTROL_DIM;

    std::shared_ptr<HyASystem> hyaSystem(new HyASystem);
    std::shared_ptr<HyASystem> hyaSystem2(new HyASystem);
    // std::shared_ptr<HyASystem> hyaSystem3(new HyASystem);


    RbdLinearizer<HyASystem> rbdLinearizer(hyaSystem, true);
    core::SystemLinearizer<STATE_DIM, CONTROL_DIM> systemLinearizer(hyaSystem2, true);
    ct::models::HyA::HyALinearizedForward hyaLinear;



    core::StateVector<STATE_DIM> x;
    x.setZero();
    core::ControlVector<CONTROL_DIM> u;
    u.setZero();

	// NumDiffComparison
    auto A_rbd = rbdLinearizer.getDerivativeState(x, u, 1.0);
    auto B_rbd = rbdLinearizer.getDerivativeControl(x, u, 1.0);


    // HyaLinearizerTest
    auto A_system = systemLinearizer.getDerivativeState(x, u, 1.0);
    auto B_system = systemLinearizer.getDerivativeControl(x, u, 1.0);

    auto A_gen = hyaLinear.getDerivativeState(x, u);

    // auto A_gen = hyaLinear.getDerivativeState(x, u, 0.0);
    // auto B_gen = hyaLinear.getDerivativeControl(x, u, 0.0);


    std::cout << "A_rbd: " << std::endl << A_rbd << std::endl << std::endl;
    std::cout << "A_system: " << std::endl << A_system << std::endl << std::endl;
    // std::cout << "A_gen: " << std::endl << A_system << std::endl << std::endl;

    std::cout << "B_rbd: " << std::endl << B_rbd << std::endl << std::endl;
    std::cout << "B_system: " << std::endl << B_system << std::endl << std::endl;
    // std::cout << "B_gen: " << std::endl << B_rbd << std::endl << std::endl;

	return 1;
}