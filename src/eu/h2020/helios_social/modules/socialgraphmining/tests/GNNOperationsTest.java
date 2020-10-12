package eu.h2020.helios_social.modules.socialgraphmining.tests;

import org.junit.Before;
import org.junit.Test;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Loss;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Matrix;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;
import org.junit.Assert;

public class GNNOperationsTest {
	@Before
	public void initializeTest() {
		Utils.development = false;
	}
	
	@Test
	public void lossShouldBeFiniteOnEdgeCases() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 1)));
	}

	@Test
	public void lossDerivativeShouldBeFiniteOnEdgeCases() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 1)));
	}

	@Test(expected = Exception.class)
	public void lossShouldThrowExceptionOnInvalidLabel() {
		Utils.development = true;
		Loss.crossEntropyDerivative(0, 0.1);
	}
	
	@Test(expected = Exception.class)
	public void lossShouldThrowExceptionOnNegativeLabel() {
		Utils.development = true;
		Loss.crossEntropyDerivative(0, -1);
	}
	
	@Test(expected = Exception.class)
	public void lossShouldThrowExceptionOnNegativeOutput() {
		Utils.development = true;
		Loss.crossEntropyDerivative(-1, 0);
	}
	
	@Test(expected = Exception.class)
	public void lossShouldThrowExceptionOnOutOfBoundsOutput() {
		Utils.development = true;
		Loss.crossEntropyDerivative(2, 0);
	}
	
	@Test
	public void tensorsShouldHaveCorrectDimensions() {
		Assert.assertEquals((new Tensor(10)).size(), 10);
	}
	
	@Test
	public void tensorShouldSerialize() {
		Tensor tensor = new Tensor(10);
		Assert.assertEquals(tensor.toString().length(), 4*10-1);
	}
	
	@Test
	public void tensorShouldBeReconstructableFromSerialization() {
		Tensor tensor = new Tensor(10);
		String originalTensor = tensor.toString();
		String newTensor = (new Tensor(originalTensor)).toString();
		Assert.assertEquals(originalTensor, newTensor);
	}
	
	@Test
	public void tensorRandomizeShouldSetNewValues() {
		Tensor tensor = new Tensor(10);
		String zeroString = tensor.toString();
		tensor.setToRandom();
		Assert.assertTrue(!zeroString.equals(tensor.toString()));
	}
	
	@Test
	public void tensorZeroCopyShouldCreateNewZeroTensor() {
		Tensor tensor = (new Tensor(10)).setToRandom();
		String originalTensor = tensor.toString();
		tensor.zeroCopy();
		Assert.assertEquals(originalTensor, tensor.toString());
	}
	
	@Test
	public void tensorMultiplicationWithZeroShouldBeZero() {
		Tensor tensor = new Tensor(10);
		String zeroString = tensor.toString();
		tensor.setToRandom().selfMultiply(0);
		Assert.assertEquals(tensor.toString(), zeroString);
	}
	
	@Test
	public void tensorSelfOperationsShouldYieldSelf() {
		Tensor tensor = new Tensor(10);
		Assert.assertSame(tensor.setToNormalized(), tensor);
		Assert.assertSame(tensor.setToRandom(), tensor);
		Assert.assertSame(tensor.setToOnes(), tensor);
		Assert.assertSame(tensor.setToUniform(), tensor);
		Assert.assertSame(tensor.setToZero(), tensor);
		Assert.assertSame(tensor.selfAdd(new Tensor(10)), tensor);
		Assert.assertSame(tensor.selfMultiply(new Tensor(10)), tensor);
		Assert.assertSame(tensor.selfSubtract(new Tensor(10)), tensor);
		Assert.assertSame(tensor.selfMultiply(0), tensor);
	}
	
	@Test
	public void tensorPairOperationsShouldYieldNewTensor() {
		Tensor tensor = new Tensor(10);
		Assert.assertNotSame(tensor.normalized(), tensor);
		Assert.assertNotSame(tensor.zeroCopy(), tensor);
		Assert.assertNotSame(tensor.add(new Tensor(10)), tensor);
		Assert.assertNotSame(tensor.multiply(new Tensor(10)), tensor);
		Assert.assertNotSame(tensor.subtract(new Tensor(10)), tensor);
		Assert.assertNotSame(tensor.multiply(0), tensor);
	}

	@Test
	public void matrixTransformShouldWorkCorrectly() {
		Matrix matrix = new Matrix(3, 2)
				.put(0, 0, 7)
				.put(1, 1, 1)
				.put(2, 1, 3);
		Tensor tensor = new Tensor(2);
		tensor.put(0, 1);
		tensor.put(1, 2);
		Tensor transformed = matrix.transform(tensor);
		double[] desired = {7., 2., 6.};
		Assert.assertArrayEquals(transformed.toArray(), desired, 0);
	}
	
	@Test
	public void zeroTensorShouldBeNormalizeable() {
		(new Tensor(10)).normalized();
	}
}
