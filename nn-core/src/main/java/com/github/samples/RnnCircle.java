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
 * @since 06.01.2015
 */
public class RnnCircle {

    private static float[][] inputData;
    private static float[][] outputData;
    private static final int size = 50;

    public static void main(String[] args) {
        Environment.getInstance().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
        Environment.getInstance().setUseWeightsSharedMemory(false);
        Environment.getInstance().setUseDataSharedMemory(false);

        NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[]{2, 3, 2}, true);

        fillData();
//        printData();

        // create training and testing input providers
        SimpleInputProvider input = new SimpleInputProvider(inputData, outputData);

        // create backpropagation trainer for the network
        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, input, input, new CustomOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.9f, 0f, 0f, 0f, 1, 1, 1000);

        // add logging
        bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

        // train
        bpt.train();

        // test
        float[][] testResult = bpt.testWithResult();
        Visualizer.createChart(outputData, inputData, testResult, false);
    }

    private static void fillData() {
        float t = -2;
        float step = (float) (Math.PI / size);

        outputData = new float[size][2];
        inputData = new float[size][2];
        for (int i = 0; i < size; i++) {
            t += step;

            outputData[i][0] = (float) ((Math.cos(2 * t) + Math.sin(2 * t)+1.8)/3.6);
            outputData[i][1] = (float) ((Math.sin(2 * t) - Math.cos(2 * t) + 1.7)/3.5);

            inputData[i][0] = createGaussianNoise(0.0000001f, outputData[i][0]);
            inputData[i][1] = createGaussianNoise(0.0000001f, outputData[i][1]);
        }

    }

    private static void printData() {
        System.out.println("====================Input data========================");
        for (int i = 0; i < inputData.length; i++) {
            System.out.println(inputData[i][0] + " " + inputData[i][1]);
        }
        System.out.println("====================Output data========================");
        for (int i = 0; i < outputData.length; i++) {
            System.out.println(outputData[i][0] + " " + outputData[i][1]);
        }
    }

    public static float createGaussianNoise(float variance, float mean) {
        Random random = new Random();
        float noise = (float) (random.nextGaussian() * Math.sqrt(variance) + mean);
        return noise;
    }
}
