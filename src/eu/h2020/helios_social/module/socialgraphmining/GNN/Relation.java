package eu.h2020.helios_social.module.socialgraphmining.GNN;

/**
 * This class provides a native java implementation of an embed-and-relate relation that
 * embeds two inputs in a lower dimensional space and then performs a triple dot product between
 * those embeddings and a vector of parameters. The output is activated through the sigmoid function.
 * 
 * @author Emmanouil Krasanakis
 */
public class Relation {
	private MatrixWithTape matrix;
	private Tensor relation;
	private Tensor relationDerivative;
	private int inputDims;
	private int accumulationSize = 0;
	
	public Relation(int inputDims, int embeddingDims) {
		this.inputDims = inputDims;
		relation = new Tensor(embeddingDims);
		matrix = new MatrixWithTape(inputDims, embeddingDims);
		matrix.setRandom();
		relation.setOnes();
	}
	private double activate(Tensor input1, Tensor input2) {
		input1.assertSize(inputDims);
		input2.assertSize(inputDims);
		return relation.dot(matrix.multiply(input1).multiply(matrix.multiply(input2)));
	}
	public double predict(Tensor input1, Tensor input2) {
		return 1./(1+Math.exp(-activate(input1, input2)));
	}
	public void startTraining() {
		matrix.startTape();
		relationDerivative = relation.zeroCopy();
		accumulationSize = 0;
	}
	public Tensor multiply(Tensor ego) {
		Tensor res = matrix.multiply(ego);
		res.setNormalize();
		return res;
	}
	public void accumulateCrossEntropy(Tensor input1, Tensor input2, double target, double weight) {
		startTraining();
		double activation = activate(input1, input2);
		double output = 1./(1+Math.exp(-activation));
		double partialEntropy = (1-target)*output - target*(1-output); // D entropy / D activation
		partialEntropy *= weight;
		System.out.println(partialEntropy*input2.norm()+" "+output+" "+activation+" "+target);
		matrix.accumulateError(input1, matrix.multiply(input2).multiply(relation).multiply(partialEntropy));
		matrix.accumulateError(input2, matrix.multiply(input1).multiply(relation).multiply(partialEntropy));
		relationDerivative.add(matrix.multiply(input1).multiply(matrix.multiply(input2)).multiply(partialEntropy));
		accumulationSize += 1;
		train();
	}
	public void train() {
		if(accumulationSize!=0)
			matrix.train(0.1, 0);
		//relation.add(relationDerivative.multiply(-0.001));
		//relation.add(relation.multiply(-0.1));
		relationDerivative = null;
		accumulationSize = 0;
	}
}
