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

using namespace ct;
using namespace ct::rbd;
using namespace ct::models::HyQ;


int main(int argc, char** argv)

{
	// obtain the state dimension
	const size_t NSTATE = ct::rbd::HyQ::Dynamics::NSTATE;
	std::cout << "NSTATE: " << NSTATE << std::endl;

	// create an instance of the system
	std::shared_ptr<ct::core::System<NSTATE>> dynamics(new ct::rbd::FloatingBaseFDSystem<ct::rbd::HyQ::Dynamics>);

	ct::core::Integrator<NSTATE> integrator(dynamics, ct::core::IntegrationType::RK4);

	ct ::core::StateVector<NSTATE> state;
	state.setZero();
	
	// simulate 1000 steps
    double dt = 0.001;
    ct::core::Time t0 = 0.0;
    size_t nSteps = 1000;
    integrator.integrate_n_steps(state, t0, nSteps, dt);


    // print the new state
    std::cout << "HyQ state after integration: " << state.transpose() << std::endl;


    typedef FloatingBaseFDSystem<HyQ::Dynamics> HyQSystem;

    const size_t STATE_DIM = HyQSystem::STATE_DIM;
    const size_t CONTROL_DIM = HyQSystem::CONTROL_DIM;

    std::shared_ptr<HyQSystem> hyqSystem(new HyQSystem);
    std::shared_ptr<HyQSystem> hyqSystem2(new HyQSystem);

    RbdLinearizer<HyQSystem> rbdLinearizer(hyqSystem, true);
    core::SystemLinearizer<STATE_DIM, CONTROL_DIM> systemLinearizer(hyqSystem2, true);
    ct::models::HyQ::HyQWithContactModelLinearizedForward hyqLinear;

    std::cout << "Testing linearisation (rbdl and system lineariser): " << std::endl;

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

    auto A_gen = hyqLinear.getDerivativeState(x, u, 0.0);
    auto B_gen = hyqLinear.getDerivativeControl(x, u, 0.0);


    std::cout << "A_rbd: " << std::endl << A_rbd << std::endl << std::endl;
    std::cout << "A_system: " << std::endl << A_system << std::endl << std::endl;
    std::cout << "A_gen: " << std::endl << A_system << std::endl << std::endl;

    std::cout << "B_rbd: " << std::endl << B_rbd << std::endl << std::endl;
    std::cout << "B_system: " << std::endl << B_system << std::endl << std::endl;
    std::cout << "B_gen: " << std::endl << B_rbd << std::endl << std::endl;


	return 1;
}