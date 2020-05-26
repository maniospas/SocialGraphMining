package eu.h2020.helios_social.module.socialgraphmining.GNN.operations;

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
	 */
	public void setToRandom() {
		for(int i=0;i<size();i++)
			put(i, Math.random());
	}
	
	/**
	 * Assign a value to a tensor element. All tensor operations use this function to wrap
	 * element assignments.
	 * @param pos The position of the tensor element
	 * @param value The value to assign
	 * @throws RuntimeException If the value is NaN or the element position is less than 0 or greater than {@link #size()}-1.
	 */
	public void put(int pos, double value) {
		if(Double.isNaN(value))
			Utils.error("Cannot accept NaN tensor values");
		else if(pos<0 || pos>=values.length)
			Utils.error("Tensor position "+pos+" out of range [0, "+values.length+")");
		else
			values[pos] = value;
	}
	/**
	 * Retrieves the value of a tensor element at a given position. All tensor operations use this function to wrap
	 * element retrieval.
	 * @param pos The position of the tensor element
	 * @return The value of the tensor element
	 * @throws RuntimeException If the element position is less than 0 or greater than {@link #size()}-1.
	 */
	public double get(int pos) {
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
	public void putAdd(int pos, double value) {
		put(pos, get(pos)+value);
	}
	/**
	 * @return The number of tensor elements
	 */
	public int size() {
		return values.length;
	}
	/**
	 * Asserts that the tensor's {@link #size()} matches the given size.
	 * @param size The size the tensor should match
	 * @throws RuntimeException if the tensor does not match the given size
	 */
	protected void assertSize(int size) {
		if(size()!=size)
			throw new RuntimeException("Different sizes: given "+size+" vs "+size());
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
	public Tensor add(Tensor tensor) {
		assertSize(tensor.size());
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The tensor to subtract
	 * @return a new Tensor that stores the outcome of subtraction
	 */
	public Tensor subtract(Tensor tensor) {
		assertSize(tensor.size());
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)-tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The tensor to perform element-wise multiplication with
	 * @return a new Tensor that stores the outcome of the multiplication
	 */
	public Tensor multiply(Tensor tensor) {
		assertSize(tensor.size());
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*tensor.get(i));
		return res;
	}
	/**
	 * @param value A number to multiply all tensor elements with
	 * @return a new Tensor that stores the outcome of the multiplication
	 */
	public Tensor multiply(double value) {
		Tensor res = zeroCopy();
		for(int i=0;i<values.length;i++)
			res.put(i, get(i)*value);
		return res;
	}
	/**
	 * Performs the dot product between this and another tensor
	 * @param tensor The tensor with which to find the product
	 * @return The dot product between the tensors.
	 */
	public double dot(Tensor tensor) {
		assertSize(tensor.size());
		double res = 0;
		for(int i=0;i<values.length;i++)
			res += get(i)*tensor.get(i);
		return res;
	}
	/**
	 * Performs the triple dot product between this and two other tensors
	 * @param tensor1 The firth other tensor with which to find the product
	 * @param tensor2 The second other tensor with which to find the product
	 * @return The triple dot product between the tensors.
	 */
	public double dot(Tensor tensor1, Tensor tensor2) {
		assertSize(tensor1.size());
		assertSize(tensor2.size());
		double res = 0;
		for(int i=0;i<values.length;i++)
			res += get(i)*tensor1.get(i)*tensor2.get(i);
		return res;
	}
	/**
	 * @return The L2 norm of the tensor
	 */
	public double norm() {
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
	 * @return A copy of the tensor on which L2 normalization has been performed
	 * @see #setToNormalized()
	 */
	public Tensor normalized() {
		double norm = norm();
		Tensor res = zeroCopy();
		if(norm!=0)
			for(int i=0;i<values.length;i++)
				res.put(i, get(i)/norm);
		return res;
	}
	/**
	 * L2-normalizes the tensor elements
	 * @see #normalized()
	 */
	public void setToNormalized() {
		double norm = norm();
		if(norm!=0)
			for(int i=0;i<values.length;i++)
				put(i, get(i)/norm);
	}
	/**
	 * Set all tensor element values to 1/{@link #size()}
	 */
	public void setToUniform() {
		for(int i=0;i<values.length;i++)
			put(i, 1./values.length);
	}
	/**
	 * Set all tensor element values to 1
	 */
	public void setToOnes() {
		for(int i=0;i<values.length;i++)
			put(i, 1.);
	}
	/**
	 * Set all tensor element values to 0
	 */
	public void setToZero() {
		for(int i=0;i<values.length;i++)
			put(i, 0.);
	}


}
