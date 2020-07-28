package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;

/**
 * Provides computation and (partial) derivation of activation and cross-entropy loss functions.
 * 
 * @author Emmanouil Krasanakis
 */
public class Loss {
	
	/**
	 * The tanh activation (exp(x)-exp(-x))/(exp(x)+exp(-x))
	 * @param x  The activation of the tanh function.
	 * @return The tanh value.
	 */
	public static double tanh(double x) {
		return (1-Math.exp(-2*x))/(1+Math.exp(-2*x));
	}
	
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
	 * The derivative of the {@link #tanh(double)} function.
	 * @param x The activation of the tanh function.
	 * @return The tanh derivative's value..
	 */
	public static double tanhDerivative(double x) {
		double tanhValue = tanh(x);
		return 1-tanhValue*tanhValue;
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
	 * @return The cross entropy partial derivative with respect to the activation passed to an intermediate sigmoid transformation.
	 */
	public static double crossEntropySigmoidDerivative(double x, double label) {
		if(label!=0 && label!=1)
			Utils.error("Only binary labels are allowed for computing the cross entropy loss");
		double sigma = sigmoid(x);
		return -label*(1-sigma) + (1-label)*sigma;
	}
	

	/**
	 * The derivative of <code>crossEntropy(tanh(x), label)</code> with respect to x. This function calculates
	 * <code>crossEntropyDerivative(tanh(x), label)*tanhDerivative(x)</code>.
	 * @param x The activation of the tanh function.
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy partial derivative with respect to the activation passed to an intermediate tanh transformation.
	 */
	public static double crossEntropyTanhDerivative(double x, double label) {
		double tanhValue = tanh(x);
		return crossEntropyDerivative(tanhValue, label)*(1-tanhValue*tanhValue);
	}
	

	
	public static Tensor sigmoid(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(int i=0;i<x.size();i++)
			ret.put(i, sigmoid(x.get(i)));
		return ret;
	}
	
	public static Tensor tanh(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(int i=0;i<x.size();i++)
			ret.put(i, tanh(x.get(i)));
		return ret;
	}
	
	public static Tensor sigmoidDerivative(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(int i=0;i<x.size();i++)
			ret.put(i, sigmoidDerivative(x.get(i)));
		return ret;
	}

	public static Tensor tanhDerivative(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(int i=0;i<x.size();i++)
			ret.put(i, tanhDerivative(x.get(i)));
		return ret;
	}
}
