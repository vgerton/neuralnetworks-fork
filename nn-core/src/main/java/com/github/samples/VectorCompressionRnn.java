package com.github.samples;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;

import java.util.Random;

/**
 * @author kisel.nikolay@gmail.com
 * @since 08.01.2015
 */
public class VectorCompressionRnn {

    private static float[][] inputData;
    private static float[][] outputData;

    public static void main(String[] args) {
        Environment.getInstance().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
        Environment.getInstance().setUseWeightsSharedMemory(false);
        Environment.getInstance().setUseDataSharedMemory(false);

        NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[]{10, 3, 10}, true);

        fillData();

        // create training and testing input providers
        SimpleInputProvider input = new SimpleInputProvider(inputData, outputData);

        // create backpropagation trainer for the network
        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new CustomOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.9f, 0.9f, 0f, 0f, 0f, 1, 1, 10000);

        // add logging
        bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

        System.out.println("Input data for learning:");
        printData(inputData);

        System.out.println("Target data for learning:");
        printData(outputData);

        // train
        bpt.train();

        // test
        float[][] testResult = bpt.testWithResult();
        printData(testResult);

        bpt.setTestingInputProvider(createTestingInputProvider());
        testResult = bpt.testWithResult();
        printData(testResult);
    }

    private static void fillData() {
        outputData = new float[][]{
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

        };

        inputData = new float[4][10];
        for (int i=0; i<outputData.length; i++) {
            for (int j=0; j<outputData[i].length; j++) {
                inputData[i][j]= createGaussianNoise(0.001f, outputData[i][j]);
            }
        }
//        inputData = new float[][]{
//                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
//                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
//
//        };

    }

    private static SimpleInputProvider createTestingInputProvider() {
        float [][] testingOutputData = new float[][]{
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

        };

        float [][] testingInputData = new float[4][10];
        for (int i=0; i<testingOutputData.length; i++) {
            for (int j=0; j<testingOutputData[i].length; j++) {
                testingInputData[i][j]= createGaussianNoise(0.001f, testingOutputData[i][j]);
            }
        }

        System.out.println("Input testing data");
        printData(testingInputData);

        System.out.println("Target testing data");
        printData(testingOutputData);


        return new SimpleInputProvider(testingInputData, testingOutputData);
    }

    private static void printData(float [][] array) {
        for (int i=0; i<array.length; i++) {
            for (int j=0; j<array[i].length; j++) {
                System.out.print(array[i][j]+" ");
            }
            System.out.println();
        }
    }

    public static float createGaussianNoise(float variance, float mean) {
        Random random = new Random();
        float noise = (float) (random.nextGaussian() * Math.sqrt(variance) + mean);
        return noise;
    }
}
