package eu.h2020.helios_social.module.socialgraphmining.GNN;

import eu.h2020.helios_social.module.socialgraphmining.GNN.operations.Tensor;


/**
 * This class provides a storage structure that organizes an embedding and a regularization target of contextual
 * ego network nodes.
 * It is indented to be used as a dynamically created instance on nodes (which are cross module components)
 * by calling <code>node.getOrCreateInstance(GNNNodeData.class)</code> to either retrieve of create it.
 * 
 * @author Emmanouil Krasanakis
 */
public class GNNNodeData {
	private static int embeddingSize = 10;
	private static double regularizationWeight = 0.1;
	
	private Tensor embedding = null;
	private Tensor regularization = null;
	public GNNNodeData() {}
	protected void initializeIfNeeded() {
		if(embedding==null) {
			embedding = new Tensor(embeddingSize);
			regularization = new Tensor(embeddingSize);
			embedding.setToRandom();
		}
	}
	
	/**
	 * 
	 * @return The embedding representation of the node.
	 */
	public synchronized Tensor getEmbedding() {
		initializeIfNeeded();
		return embedding;
	}
	
	/**
	 * Sets the regularization (default is a zero vector) of the {@link #updateEmbedding(Tensor, double)} operation.
	 * Setting this to a value other than zero helps influence the trained embedding to a similar latent space than the 
	 * one of that value. Hence, this can be used to influence the embeddings of nodes in the contextual ego network
	 * given their embeddings in other devices.
	 * @param regularization
	 */
	public synchronized void setRegularization(Tensor regularization) {
		this.regularization = regularization;
	}
	
	/**
	 * Performs the operation <i>embedding += (embedding-regularization)*learningRate*regularizationWeight-derivative*learningRate</i>
	 * that is a regularized gradient descent over a computed derivative, where the area of regularization is constrained towards the point
	 * set by {@link #setRegularization(Tensor)}.
	 * @param derivative The derivative of the embedding.
	 * @param learningRate The learning rate.
	 * @see #getEmbedding()
	 */
	public synchronized void updateEmbedding(Tensor derivative, double learningRate) {
		embedding = embedding
						.add(regularization.subtract(embedding).multiply(regularizationWeight*learningRate))
						.add(derivative.multiply(-learningRate));
	}
};
