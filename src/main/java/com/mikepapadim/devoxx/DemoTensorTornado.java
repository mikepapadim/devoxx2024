package com.mikepapadim.devoxx;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;

public class DemoTensorTornado {
    public static void tensorAdditionFloat32(TensorFP32 tensorA, TensorFP32 tensorB, TensorFP32 tensorC) {
        for (@Parallel int i = 0; i < tensorC.getSize(); i++) {
            tensorC.set(i, tensorA.get(i) + tensorB.get(i));
        }
    }

    public static void main(String[] args) {
        // Define the shape for the tensors
        Shape shape = new Shape(16, 16);

        // Create two tensors and initialize their values
        TensorFP32 tensorA = new TensorFP32(shape);
        TensorFP32 tensorB = new TensorFP32(shape);

        // Create a tensor to store the result of addition
        TensorFP32 tensorC = new TensorFP32(shape);

        // Step 3: Initialize tensorA with the value 1.0f.
        tensorA.init(1.0f);

        // Initialize tensorB with a sequence of values.
        for (int i = 0; i < tensorB.getSize(); i++) {
            tensorB.set(i, i + 1.0f); // Setting values from 1.0f, 2.0f, ..., tensorB.getSize().
        }

        // Define the task graph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, tensorA, tensorB)
                .task("t0", DemoTensorTornado::tensorAdditionFloat32, tensorA, tensorB, tensorC)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, tensorC);

        // Take a snapshot of the task graph
        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        // Create an execution plan and execute it
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph);
        executionPlan.execute();

        DemoUtils.prettyPrintTensor(tensorC, tensorC.getShape());
    }

    
}
