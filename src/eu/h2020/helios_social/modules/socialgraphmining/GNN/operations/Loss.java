package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;

/**
 * Provides computation and (partial) derivation of activation and cross-entropy loss functions.
 * 
 * @author Emmanouil Krasanakis
 */
public class Loss {
	/**
	 * The sigmoid function 1/(1+exp(-x)).
	 * @param x The activation of the sigmoid function.
	 * @return The sigmoid value.
	 */
	public static double sigmoid(double x) {
		return 1./(1+Math.exp(-x));
	}
	
	/**
	 * The derivative of the {@link #sigmoid(double)} function.
	 * @param x The activation of the sigmoid function.
	 * @return The sigmoid derivative's value.
	 */
	public static double sigmoidDerivative(double x) {
		double sigma = sigmoid(x);
		return sigma*(1-sigma);
	}
	
	/**
	 * A cross entropy loss for one sample computes as -label*log(output) -(1-label)*log(1-output). To avoid producing invalid
	 * values, an eps of 1.E-12 is used to constraint the cross entropy in the range [-12, 12].
	 * @param output The output of a prediction task. Should lie in the range [0,1]
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy value.
	 */
	public static double crossEntropy(double output, double label) {
		if(label!=0 && label!=1)
			Utils.error("Only binary labels are allowed for computing the cross entropy loss");
		if(output<0 || output>1)
			Utils.error("The predicted output passed on to cross entropy should lie in the range [0,1]");
		return -label*Math.log(output+1.E-12) - (1-label)*Math.log(1-output+1.E-12);
	}

	/**
	 * The derivative of the {@link #crossEntropy(double, double)} loss. To avoid producing invalid
	 * values, an eps of 1.E-12 is used to constraint the cross entropy in the range [-12, 12], which results
	 * to this derivative being constrained in the range [-1.E12, 1.E12].
	 * @param output The output of a prediction task. Should lie in the range [0,1]
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy derivative's value.
	 */
	public static double crossEntropyDerivative(double output, double label) {
		if(label!=0 && label!=1)
			Utils.error("Only binary labels are allowed for computing the cross entropy loss");
		if(output<0 || output>1)
			Utils.error("The predicted output passed on to cross entropy should lie in the range [0,1]");
		return -label/(output+1.E-12) + (1-label)/(1-output+1.E-12);
	}
	
	/**
	 * The derivative of <code>crossEntropy(sigmoid(x), label)</code> with respect to x. This function can avoid
	 * using an eps and is hence more precise than the expression
	 * <code>crossEntropyDerivative(sigmoid(x), label)*sigmoidDerivative(x)</code>.
	 * @param x The activation of the sigmoid function.
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy partial derivative with respect to the activation passed to an intermedia sigmoid transformation.
	 */
	public static double crossEntropySigmoidDerivative(double x, double label) {
		double sigma = sigmoid(x);
		return -label*(1-sigma) + (1-label)*sigma;
	}
}
