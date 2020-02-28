package eu.h2020.helios_social.module.socialgraphmining.GNN;

import java.util.HashMap;

/**
 * A matrix extended with tape capabilities that can compute and store gradients over time.
 * 
 * @author Emmanouil Krasanakis
 */
public class MatrixWithTape extends Matrix {
	private Matrix tape;
	private HashMap<Tensor, Tensor> history = new HashMap<Tensor, Tensor>();
	public MatrixWithTape(int inputSize, int outputSize) {
		super(inputSize, outputSize);
		tape = null;
	}
	public Tensor lastOutput(Tensor input) {
		return history.get(input);
	}
	@Override
	public Tensor multiply(Tensor input) {
		Tensor output = super.multiply(input);
		history.put(input, output);
		return output;
	}
	/**
	 * Restarts the tape into a zero matrix.
	 */
	public void startTape() {
		tape = new Matrix(inputSize, outputSize);
	}
	/**
	 * Accumulate error function onto the matrix tape, considering as a linear transformation of an input to an output.
	 * @param input The transformation's input
	 * @param error The transformation's output
	 */
	public void accumulateError(Tensor input, Tensor error) {
		if(tape==null)
			throw new RuntimeException("Must start tape before accumulating error");
		Tensor output = history.get(input);
		if(output==null)
			throw new RuntimeException("Needs to run once before accumulating error for an input");
		for(int i=0;i<inputSize;i++)
			for(int j=0;j<outputSize;j++) 
				tape.addW(i, j, input.get(i));
	}
	/**
	 * Train on the accumulated tape
	 * @param learningRate The training learning rate
	 * @param regularization An L1 regularization term to avoid overfitting and promote sparsity
	 */
	public void train(double learningRate, double regularization) {
		if(tape==null)
			throw new RuntimeException("Must start tape and accumulate error before training");
		//tape.normalize();
		for(int i=0;i<size();i++)
			put(i, tape.get(i)*learningRate - get(i)*regularization);
		//System.out.println(tape.norm());
		history.clear();
		tape.setZero();
	}
}
