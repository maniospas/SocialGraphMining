package eu.h2020.helios_social.modules.socialgraphmining.GNN.operations;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;

public class Matrix extends Tensor {
	private int rows;
	private int cols;
	
	public Matrix(int rows, int cols) {
		super(rows*cols);
		this.rows = rows;
		this.cols = cols;
	}
	
	protected Matrix() {
	}
	
	@Override
	public Matrix zeroCopy() {
		return new Matrix(rows, cols);
	}
	
	public final int getRows() {
		return rows;
	}
	
	public final int getCols() {
		return cols;
	}
	
	public final double get(int row, int col) {
		if(row<0 || col<0 || row>=rows || col>=cols)
			Utils.error("Matrix element out of range");
		return get(row+col*rows);
	}
	
	public final Matrix put(int row, int col, double value) {
		put(row+col*rows, value);
		return this;
	}
	
	public final Matrix transposed() {
		return zeroCopy().selfTransposed(); 
	}
	
	public final Matrix selfTransposed() {
		if(rows!=cols)
			Utils.error("In-place transposition requires a square matrix");
		for(int row=0;row<rows;row++)
			for(int col=0;col<cols;col++) {
				double temp = get(row, col);
				put(row, col, get(col, row));
				put(col, row, temp);
			}
		return this;
	}
	
	/**
	 * Performs the linear algebra transformation A*x where A is this matrix and x a vector
	 * @param x The one-dimensional tensor which is the vector being transformed.
	 * @return The one-dimensional outcome of the transformation.
	 */
	public final Tensor transform(Tensor x) {
		x.assertSize(cols);
		Tensor ret = new Tensor(rows);
		for(int row=0;row<rows;row++) {
			double rowValue = 0;
			for(int col=0;col<cols;col++)
				rowValue += get(row, col)*x.get(col);
			ret.put(row, rowValue);
		}
		return ret;
	}
	
	public final Matrix matmul(Matrix with) {
		if(cols!=with.getRows())
			Utils.error("Mismatched matrix sizes");
		Matrix ret = new Matrix(getRows(), with.getCols());
		for(int row=0;row<rows;row++)
			for(int col=0;col<with.getCols();col++) {
				double elementValue = 0;
				for(int k=0;k<cols;k++)
					elementValue += get(row, k)*with.get(k, col);
				ret.put(row, col, elementValue);
			}
		return ret;
	}
	
	public static Matrix external(Tensor horizontal, Tensor vertical) {
		Matrix ret = new Matrix(horizontal.size(), vertical.size());
		for(int row=0;row<horizontal.size();row++)
			for(int col=0;col<vertical.size();col++) 
				ret.put(row, col, horizontal.get(row)*vertical.get(col));
		return ret;
	}
	
	@Override
	protected void assertMatching(Tensor other) {
		if(!(other instanceof Matrix))
			throw new RuntimeException("Non-compliant: "+describe()+" vs "+other.describe());
		if(rows!=((Matrix)other).rows || cols!=((Matrix)other).cols)
			throw new RuntimeException("Non-compliant: "+describe()+" vs "+other.describe());
	}
	
	/*@Override
	public String toString() {
		String ret = "";
		for(int row=0;row<rows;row++) {
			if(cols>0)
				ret += get(row, 0);
			for(int col=1;col<cols;col++) 
				ret += ","+get(row, col);
			ret += "\n";
		}
		return "[\n"+ret+"]";
	}*/
	
	@Override
	public String describe() {
		return "Matrix ("+rows+","+cols+")";
	}
}
