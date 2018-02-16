/**********************************************************************************************************************
This file is part of the Control Toolbox (https://adrlab.bitbucket.io/ct), copyright by ETH Zurich, Google Inc.
Authors:  Michael Neunert, Markus Giftthaler, Markus Stäuble, Diego Pardo, Farbod Farshidian
Licensed under Apache2 license (see LICENSE file in main directory)
**********************************************************************************************************************/
#include <ct/core/core.h>
#include "my_ct_project/CustomController.h"

int main(int argc, char** argv)
{
    // a damped oscillator has two states, position and velocity
    const size_t state_dim = ct::core::SecondOrderSystem::STATE_DIM;      // = 2
    const size_t control_dim = ct::core::SecondOrderSystem::CONTROL_DIM;  // = 1
    // create a state
    ct::core::StateVector<state_dim> x;
    // we initialize it at a point with unit deflection and zero velocity
    x(0) = 1.0;
    x(1) = 0.0;
    // create our oscillator
    double w_n = 70;
    std::shared_ptr<ct::core::SecondOrderSystem> oscillator(new ct::core::SecondOrderSystem(w_n));
    // create our controller
    double kp = 10;
    double kd = 1;
    ct::core::ControlVector<control_dim> uff;
    uff << 2.0;
    std::shared_ptr<CustomController> controller(new CustomController(uff, kp, kd));
    // assign our controller
    oscillator->setController(controller);
    // create an integrator
    ct::core::Integrator<state_dim> integrator(oscillator, ct::core::IntegrationType::RK4);
    // simulate 1000 steps
    double dt = 0.001;
    ct::core::Time t0 = 0.0;
    size_t nSteps = 1000;
    integrator.integrate_n_steps(x, t0, nSteps, dt);
    // print the new state
    std::cout << "state after integration:: " << x.transpose() << std::endl;
    return 1;
}