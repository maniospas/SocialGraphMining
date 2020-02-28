package eu.h2020.helios_social.module.socialgraphmining.GNN;

/**
 * Native java implementation of a dense matrix built on the {@link Tensor} data structure.
 * 
 * @author Emmanouil Krasanakis
 */
public class Matrix extends Tensor {
	protected int inputSize;
	protected int outputSize;
	/**
	 * Creates the matrix given the respective dimensions.
	 * @param inputSize
	 * @param outputSize
	 */
	public Matrix(int inputSize, int outputSize) {
		super(inputSize*outputSize);
		this.inputSize = inputSize;
		this.outputSize = outputSize;
	}
	public double getW(int inputPos, int outputPos) {
		if(outputPos>=outputSize)
			throw new RuntimeException("Output position should be less than "+outputSize);
		return get(inputPos*outputSize+outputPos);
	}
	public void putW(int inputPos, int outputPos, double value) {
		if(outputPos>=outputSize)
			throw new RuntimeException("Output position should be less than "+outputSize);
		 put(inputPos*outputSize+outputPos, value);
	}
	public void addW(int inputPos, int outputPos, double value) {
		putW(inputPos, outputPos, getW(inputPos, outputPos) + value);
	}
	@Override
	public Tensor multiply(Tensor input) {
		input.assertSize(inputSize);
		Tensor output = new Tensor(outputSize);
		for(int i=0;i<inputSize;i++)
			for(int j=0;j<outputSize;j++)
				output.putAdd(j, getW(i, j)*input.get(i));
		return output;
	}
}
