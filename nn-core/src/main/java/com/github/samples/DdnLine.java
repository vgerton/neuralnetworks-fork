package com.github.samples;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.architecture.types.DBN;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.training.OneStepTrainer;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.training.rbm.AparapiCDTrainer;
import com.github.neuralnetworks.training.rbm.DBNTrainer;
import com.github.neuralnetworks.util.Environment;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * @author kisel.nikolay@gmail.com
 * @since 06.01.2015
 */
public class DdnLine {

    private static float[][] inputData;
    private static float[][] outputData;
    private static final int size = 50;

    public static void main(String[] args) {
        Environment.getInstance().setExecutionMode(Kernel.EXECUTION_MODE.CPU);
        Environment.getInstance().setUseWeightsSharedMemory(false);
        Environment.getInstance().setUseDataSharedMemory(false);

        DBN dbn = NNFactory.dbn(new int[]{2, 3, 1, 2}, true);

        dbn.setLayerCalculator(NNFactory.lcSigmoid(dbn, null));

        fillData();
        printData();

        // create training and testing input providers
        SimpleInputProvider inputProvider = new SimpleInputProvider(inputData, outputData);

        // rbm trainers for each layer
        AparapiCDTrainer firstTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getNeuralNetwork(0), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.5f, 0f, 0f, 1, 1, 100, true);
        AparapiCDTrainer secondTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getNeuralNetwork(1), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.5f, 0f, 0f, 1, 1, 100, true);
        AparapiCDTrainer thirdTrainer = TrainerFactory.cdSigmoidTrainer(dbn.getNeuralNetwork(2), null, null, null, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.5f, 0f, 0f, 1, 1, 100, true);

        Map<NeuralNetwork, OneStepTrainer<?>> map = new HashMap<>();
        map.put(dbn.getNeuralNetwork(0), firstTrainer);
        map.put(dbn.getNeuralNetwork(1), secondTrainer);
        map.put(dbn.getNeuralNetwork(2), thirdTrainer);

        // deep trainer
        DBNTrainer deepTrainer = TrainerFactory.dbnTrainer(dbn, map, inputProvider, null, null);

        Environment.getInstance().setExecutionMode(Kernel.EXECUTION_MODE.CPU);

        // layer pre-training
        deepTrainer.train();

        // fine tuning backpropagation
        BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(dbn, inputProvider, inputProvider, new CustomOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.7f, 0f, 0f, 0f, 1, 1, 1000);

        // log data
        bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName()));

        // training
        bpt.train();

        // testing
        float [][] testResult = bpt.testWithResult();
        Visualizer.createChart(outputData, inputData, testResult, true);
    }

    private static void fillData() {
        float step = (float) (0.8 / size);
        float x = 0.10f;

        outputData = new float[size][2];
        inputData = new float[size][2];
        for (int i = 0; i < size; i++) {
            outputData[i][0] = x;
            outputData[i][1] = (float) (0.7 * x + 0.1);

            inputData[i][0] = createGaussianNoise(0.0001f, outputData[i][0]);
            inputData[i][1] = createGaussianNoise(0.0001f, outputData[i][1]);

            x += step;
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
