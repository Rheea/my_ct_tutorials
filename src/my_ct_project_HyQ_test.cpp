#include <ct/core/core.h>
#include "my_ct_project/CustomController.h"
#include <ct/models/HyQ/HyQ.h>
#include <ct/models/HyA/HyA.h>
#include <ct/rbd/systems/FixBaseFDSystem.h>
#include <ct/rbd/systems/FloatingBaseFDSystem.h>
#include <ct/models/InvertedPendulum/InvertedPendulum.h>

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
    std::cout << "state after integration: " << state.transpose() << std::endl;

	return 1;
}