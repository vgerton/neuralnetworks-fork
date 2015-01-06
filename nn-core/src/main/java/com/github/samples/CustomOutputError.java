package com.github.samples;

import com.github.neuralnetworks.calculation.OutputError;
import com.github.neuralnetworks.tensor.Tensor;

import java.util.Iterator;

/**
 * @author kisel.nikolay@gmail.com
 * @since 06.01.2015
 */
public class CustomOutputError implements OutputError {

    private static final long serialVersionUID = 1L;

    private float networkError;
    private float squareNetworkError;
    private int size;
    private int errorSamples;

    @Override
    public void addItem(Tensor networkOutput, Tensor targetOutput) {
        Iterator<Integer> targetIt = targetOutput.iterator();
        Iterator<Integer> actualIt = networkOutput.iterator();
        size += targetOutput.getDimensions()[targetOutput.getDimensions().length - 1];
        float error = 0;
        float squareError = 0;
        while (targetIt.hasNext() && actualIt.hasNext()) {
            error += Math.abs(Math.abs(networkOutput.getElements()[actualIt.next()]) - Math.abs(targetOutput.getElements()[targetIt.next()]));
            squareError = error * error;
        }
        networkError += error;
        squareNetworkError += squareError;
        if (error / targetOutput.getDimensions()[targetOutput.getDimensions().length - 1] > 0.3) {
            errorSamples += targetOutput.getDimensions()[targetOutput.getDimensions().length - 1];
        }
    }

    @Override
    public float getTotalNetworkError() {
        return size > 0 ? networkError / size : 0;
    }

    @Override
    public float getTotalSquareNetworkError() {
        return size > 0 ? squareNetworkError / 2 : 0;
    }

    @Override
    public int getTotalErrorSamples() {
        return errorSamples;
    }

    @Override
    public int getTotalInputSize() {
        return size;
    }

    @Override
    public void reset() {
        networkError = 0;
        errorSamples = 0;
        squareNetworkError = 0;
        size = 0;
    }
}
