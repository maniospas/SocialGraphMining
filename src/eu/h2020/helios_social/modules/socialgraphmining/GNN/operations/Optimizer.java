package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Serializer.Serialization;

/**
 * Provides an interface for training tensors. Has a {@link #reset()} method that starts potential training memory from scratch.
 * Has an {@link #update(Tensor, Tensor)} method that, given a current value and a gradient Tensor suggests a new value.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract interface Optimizer {
	public void update(Tensor value, Tensor gradient);
	public default void reset() {};
	
	public class Regularization implements Optimizer {
		private Optimizer baseRule;
		protected double regularization;
		public Regularization(Optimizer baseRule, double regularization) {
			this.baseRule = baseRule;
			this.regularization = regularization;
		}
		protected Regularization() {}
		@Override
		public void update(Tensor value, Tensor gradient) {
			if(regularization==0)
				baseRule.update(value, gradient);
			else
				baseRule.update(value, gradient.add(value.multiply(regularization)));
		}
		@Override
		public void reset() {
			baseRule.reset();
		}
	}
	
	public class GradientDescent implements Optimizer {
		protected double learningRate;
		public GradientDescent(double learningRate) {
			this.learningRate = learningRate;
		}
		protected GradientDescent() {}
		@Override
		public void update(Tensor value, Tensor gradient) {
			value.selfAdd(gradient.multiply(-learningRate));
		}
	}
	
	public class Adam implements Optimizer {
		private double b1;
		private double b2;
		private double learningRate;
		private static double espilon = 0.0001;
		
		@Serialization(enabled=false)
		private HashMap<Tensor, Tensor> m = new HashMap<Tensor, Tensor>();
		@Serialization(enabled=false)
		private HashMap<Tensor, Tensor> v = new HashMap<Tensor, Tensor>();
		
		public Adam(double learningRate) {
			this(learningRate, 0.9, 0.999);
		}
		public Adam(double learningRate, double b1, double b2) {
			this.learningRate = learningRate;
			this.b1 = b1;
			this.b2 = b2;
		}
		protected Adam() {}
		@Override
		public void update(Tensor value, Tensor gradient) {
			if(!m.containsKey(value))
				m.put(value, value.zeroCopy());
			if(!v.containsKey(value))
				v.put(value, value.zeroCopy());
			
			m.get(value).selfMultiply(b1).selfAdd(gradient.multiply(1-b1));
			v.get(value).selfMultiply(b2).selfAdd(gradient.multiply(gradient).selfMultiply(1-b2));
			
			Tensor mHat = m.get(value).multiply(1./(1-b1));
			Tensor vHat = m.get(value).multiply(1./(1-b2));
			value.selfAdd(mHat.selfMultiply(-learningRate).selfMultiply(vHat.selfSqrt().selfAdd(espilon).selfInverse()));
		}
		@Override
		public void reset() {
			m = new HashMap<Tensor, Tensor>();
			v = new HashMap<Tensor, Tensor>();
		}
	}
}
