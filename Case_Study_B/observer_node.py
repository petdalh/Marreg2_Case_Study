#!/usr/bin/env python3
#
# This file is part of CyberShip Enterpries Suite.
#
# CyberShip Enterpries Suite software is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CyberShip Enterpries Suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# CyberShip Enterpries Suite. If not, see <https://www.gnu.org/licenses/>.
#
# Maintainer: Emir Cem Gezer
# Email: emir.cem.gezer@ntnu.no, emircem.gezer@gmail.com, me@emircem.com
# Year: 2022
# Copyright (C) 2024 NTNU Marine Cybernetics Laboratory

import rclpy
import rclpy.node
import numpy as np
import rcl_interfaces.msg

import std_msgs.msg
import tmr4243_interfaces.msg

from template_observer.luenberger import luenberger
from template_observer.wrap import wrap


class Observer(rclpy.node.Node):
    TASK_DEADRECKONING = 'deadreckoning'
    TASK_LUENBERGER = 'luenberger'
    TASK_LIST = [TASK_DEADRECKONING, TASK_LUENBERGER]

    def __init__(self):
        super().__init__('cse_observer')

        self.subs = {}
        self.pubs = {}

        self.subs["tau"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/tau', self.tau_callback, 10
        )
        self.subs["eta"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10
        )
        self.pubs['observer'] = self.create_publisher(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/eta', 1
        )

        self.task = Observer.TASK_LUENBERGER
        self.declare_parameter(
            'task',
            self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Task",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                read_only=False,
                additional_constraints=f"Allowed values: {' '.join(Observer.TASK_LIST)}"
            )
        )
        self.task = self.get_parameter('task').get_parameter_value().string_value

        self.L1_value = [1.0] * 3
        self.declare_parameter(
            'L1',
            self.L1_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L1 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L2_value = [1.0] * 3
        self.declare_parameter(
            'L2',
            self.L2_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L2 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L3_value = [1.0] * 3
        self.declare_parameter(
            'L3',
            self.L3_value,
            rcl_interfaces.msg.ParameterDescriptor(
                description="L3 gain",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        self.L1_value = self.get_parameter('L1').get_parameter_value().double_array_value
        self.L2_value = self.get_parameter('L2').get_parameter_value().double_array_value
        self.L3_value = self.get_parameter('L3').get_parameter_value().double_array_value

        self.get_logger().info(f"Task: {self.task}")

        self.last_eta = np.zeros((3, 1), dtype=float)

        self.last_tau = np.zeros((3, 1), dtype=float)

        self.observer_runner = self.create_timer(0.1, self.observer_loop)

    def observer_loop(self):

        L1 = np.diag(self.L1_value)
        L2 = np.diag(self.L2_value)
        L3 = np.diag(self.L3_value)

        eta_hat, nu_hat, bias_hat = luenberger(
            self.last_eta,
            self.last_tau,
            L1,
            L2,
            L3
        )

        obs = tmr4243_interfaces.msg.Observer()
        obs.eta = eta_hat.flatten().tolist()
        obs.nu = nu_hat.flatten().tolist()
        obs.bias = bias_hat.flatten().tolist()
        self.pubs['observer'].publish(obs)

    def tau_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_tau = np.array([msg.data], dtype=float).T

    def eta_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_eta = np.array([msg.data], dtype=float).T


def main():
    rclpy.init()

    node = Observer()

    rclpy.spin(node)


if __name__ == '__main__':
    main()