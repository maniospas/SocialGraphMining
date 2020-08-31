package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;

/**
 * This class provides a native java implementation of Tensor functionalities.
 * 
 * @author Emmanouil Krasanakis
 */
public class Tensor {
	private double[] values;
	/**
	 * Constructor that reconstructs a serialized Tensor (i.e. the outcome of {@link #toString()})
	 * @param expr A serialized tensor
	 */
	public Tensor(String expr) {
		if(expr==null || expr.isEmpty())
			Utils.error("Cannot create tensor from a null expression or empty string");
		if(expr.length()==0) {
			values = new double[0];
			return;
		}
		String[] splt = expr.split(",");
		values = new double[splt.length];
		for(int i=0;i<splt.length;i++)
			put(i, Double.parseDouble(splt[i]));
	}
	/**
	 * Construct that creates a tensor of zeros given its number of elements
	 * @param size The number of tensor elements
	 */
	public Tensor(int size) {
		values = new double[size];
	}
	protected Tensor() {}
	/**
	 * Set tensor elements to random values in the range [0,1]
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToRandom() {
		for(int i=0;i<size();i++)
			put(i, Math.random());
		return this;
	}
	/**
	 * Assign a value to a tensor element. All tensor operations use this function to wrap
	 * element assignments.
	 * @param pos The position of the tensor element
	 * @param value The value to assign
	 * @throws RuntimeException If the value is NaN or the element position is less than 0 or greater than {@link #size()}-1.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor put(int pos, double value) {
		if(Double.isNaN(value))
			Utils.error("Cannot accept NaN tensor values");
		else if(pos<0 || pos>=values.length)
			Utils.error("Tensor position "+pos+" out of range [0, "+values.length+")");
		else
			values[pos] = value;
		return this;
	}
	/**
	 * Retrieves the value of a tensor element at a given position. All tensor operations use this function to wrap
	 * element retrieval.
	 * @param pos The position of the tensor element
	 * @return The value of the tensor element
	 * @throws RuntimeException If the element position is less than 0 or greater than {@link #size()}-1.
	 */
	public final double get(int pos) {
		if(pos<0 || pos>=values.length)
			return Utils.error("Tensor position "+pos+" out of range [0, "+values.length+")", 0);
		return values[pos];
	}
	/**
	 * Add a value to a tensor element.
	 * @param pos The position of the tensor element
	 * @param value The value to assign
	 * @see #put(int, double)
	 */
	public final void putAdd(int pos, double value) {
		put(pos, get(pos)+value);
	}
	/**
	 * @return The number of tensor elements
	 */
	public final int size() {
		return values.length;
	}
	/**
	 * Asserts that the tensor's {@link #size()} matches the given size.
	 * @param size The size the tensor should match
	 * @throws RuntimeException if the tensor does not match the given size
	 */
	protected final void assertSize(int size) {
		if(size()!=size)
			throw new RuntimeException("Different sizes: given "+size+" vs "+size());
	}
	/**
	 * Asserts that the tensor's dimensions match with another tensor. This check can be made
	 * more complex by derived classes, but for a base Tensor instance it calls {@link #assertSize(int)}.
	 * @param other The other tensor to compare with.
	 */
	protected void assertMatching(Tensor other) {
		assertSize(other.size());
	}
	/**
	 * @return A tensor with the same size but zero elements
	 */
	public Tensor zeroCopy() {
		return new Tensor(values.length);
	}
	/**
	 * @param tensor The tensor to add with
	 * @return a new Tensor that stores the outcome of addition
	 */
	public final Tensor add(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The value to add to each element
	 * @return a new Tensor that stores the outcome of addition
	 */
	public final Tensor add(double value) {
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)+value);
		return res;
	}
	/**
	 * Performs in-memory addition to the Tensor, storing the result in itself.
	 * @param tensor The tensor to add (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfAdd(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory addition to the Tensor, storing the result in itself.
	 * @param tensor The value to add to each tensor element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfAdd(double value) {
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)+value);
		return res;
	}
	/**
	 * @param tensor The tensor to subtract
	 * @return a new Tensor that stores the outcome of subtraction
	 */
	public final Tensor subtract(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)-tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory subtraction from the Tensor, storing the result in itself.
	 * @param tensor The tensor to subtract (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfSubtract(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)-tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The tensor to perform element-wise multiplication with.
	 * @return A new Tensor that stores the outcome of the multiplication.
	 */
	public final Tensor multiply(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory multiplication on the Tensor, storing the result in itself .
	 * @param tensor The tensor to perform element-wise multiplication with  (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfMultiply(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*tensor.get(i));
		return res;
	}
	/**
	 * @param value A number to multiply all tensor elements with.
	 * @return A new Tensor that stores the outcome of the multiplication.
	 */
	public final Tensor multiply(double value) {
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*value);
		return res;
	}
	/**
	 * Performs in-memory multiplication on the Tensor, storing the result to itself.
	 * @param value A number to multiply all tensor elements with.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfMultiply(double value) {
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*value);
		return res;
	}
	/**
	 * @return A new Tensor that stores the outcome of finding the absolute square root of each element.
	 */
	public final Tensor sqrt() {
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, Math.sqrt(Math.abs(get(i))));
		return res;
	}
	/**
	 * Performs in-memory the square root of the absolute of each element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfSqrt() {
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			res.put(i, Math.sqrt(Math.abs(get(i))));
		return res;
	}
	/**
	 * @return A new Tensor with inversed each non-zero element.
	 */
	public final Tensor inverse() {
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			if(get(i)!=0)
				res.put(i, 1./get(i));
		return res;
	}
	/**
	 * Performs in-memory the inverse of each non-zero element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfInverse() {
		Tensor res = this;
		for(int i=0;i<values.length;i++)
			if(get(i)!=0)
				res.put(i, 1./get(i));
		return res;
	}
	/**
	 * Performs the dot product between this and another tensor.
	 * @param tensor The tensor with which to find the product.
	 * @return The dot product between the tensors.
	 */
	public final double dot(Tensor tensor) {
		assertMatching(tensor);
		double res = 0;
		for(int i=0;i<values.length;i++)
			res += get(i)*tensor.get(i);
		return res;
	}
	/**
	 * Performs the triple dot product between this and two other tensors.
	 * @param tensor1 The firth other tensor with which to find the product.
	 * @param tensor2 The second other tensor with which to find the product.
	 * @return The triple dot product between the tensors.
	 */
	public final double dot(Tensor tensor1, Tensor tensor2) {
		assertMatching(tensor1);
		assertMatching(tensor2);
		double res = 0;
		for(int i=0;i<values.length;i++)
			res += get(i)*tensor1.get(i)*tensor2.get(i);
		return res;
	}
	/**
	 * @return The L2 norm of the tensor
	 */
	public final double norm() {
		double res = 0;
		for(double value : values)
			res += value*value;
		return Math.sqrt(res);
	}
	/**
	 * A string serialization of the tensor that can be used by the constructor {@link #Tensor(String)} to create an identical copy.
	 * @return A serialization of the tensor.
	 */
	@Override
	public String toString() {
		StringBuilder res = new StringBuilder();
		if(size()!=0)
			res.append(get(0));
		for(int i=1;i<size();i++)
			res.append(",").append(get(i));
		return res.toString();
	}
	/**
	 * @return A copy of the tensor on which L2 normalization has been performed.
	 * @see #setToNormalized()
	 */
	public final Tensor normalized() {
		double norm = norm();
		Tensor res = zeroCopy();
		if(norm!=0)
			for(int i=0;i<values.length;i++)
				res.put(i, get(i)/norm);
		return res;
	}
	/**
	 * L2-normalizes the tensor's elements.
	 * @return <code>this</code> Tensor instance.
	 * @see #normalized()
	 */
	public final Tensor setToNormalized() {
		double norm = norm();
		if(norm!=0)
			for(int i=0;i<values.length;i++)
				put(i, get(i)/norm);
		return this;
	}
	/**
	 * Set all tensor element values to 1/{@link #size()}
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToUniform() {
		for(int i=0;i<values.length;i++)
			put(i, 1./values.length);
		return this;
	}
	/**
	 * Set all tensor element values to 1.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToOnes() {
		for(int i=0;i<values.length;i++)
			put(i, 1.);
		return this;
	}
	/**
	 * Set all tensor element values to 0.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToZero() {
		for(int i=0;i<values.length;i++)
			put(i, 0.);
		return this;
	}
	/**
	 * Retrieves a representation of the Tensor as an array of doubles.
	 * @return An array of doubles
	 */
	public final double[] toArray() {
		double[] values = new double[this.values.length];
		for(int i=0;i<this.values.length;i++)
			values[i] = this.values[i];
		return values;
	}
	
	public static Tensor fromDouble(double value) {
		Tensor ret = new Tensor(1);
		ret.put(0, value);
		return ret;
	}
	
	public double toDouble() {
		assertSize(1);
		return get(0);
	}
	public String describe() {
		return "Tensor ("+size()+")";
	}

}
