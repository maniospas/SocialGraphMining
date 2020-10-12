package eu.h2020.helios_social.modules.socialgraphmining.GNN;

import java.util.ArrayList;
import java.util.LinkedList;

import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;


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
	private double learningRate = 1;
	private double regularizationWeight = 0.1;
	
	private Tensor embedding = null;
	private Tensor regularization = null;
	private Tensor neighborAggregation = null;
	
	private LinkedList<Tensor> embeddingHistory = null;
	
	public GNNNodeData() {}
	protected void initializeIfNeeded() {
		if(embedding==null) {
			embedding = new Tensor(embeddingSize);
			regularization = new Tensor(embeddingSize);
			neighborAggregation = new Tensor(embeddingSize);
			embeddingHistory = new LinkedList<Tensor>();
			embedding.setToRandom();
		}
	}
	
	public synchronized void addEmbeddingToHistory() {
		initializeIfNeeded();
		if(embeddingHistory.size()>10)
			embeddingHistory.remove(0);
		embeddingHistory.add(embedding.add(0));
	}
	
	public synchronized void addRegularizationToHistory() {
		initializeIfNeeded();
		if(embeddingHistory.size()>100)
			embeddingHistory.removeFirst();
		embeddingHistory.add(regularization.add(0));
	}
	
	public synchronized LinkedList<Tensor> getEmbeddingHistory() {
		initializeIfNeeded();
		return embeddingHistory;
	}
	
	/**
	 * Retrieves the embedding of the node.
	 * @return A Tensor holding the embedding representation.
	 */
	public synchronized Tensor getEmbedding() {
		initializeIfNeeded();
		return embedding;
	}
	
	/**
	 * Forcibly sets an embedding tensor. Normally, embeddings are updated through {@link #updateEmbedding(Tensor)}.
	 * @param embedding The embedding Tensor
	 */
	synchronized void forceSetEmbedding(Tensor embedding) {
		initializeIfNeeded();
		this.embedding = embedding;
	}
	
	/**
	 * Sets a neighbor aggregation that can be retrieved with {@link #getNeighborAggregation()}. This
	 * aggregation is computed by other devices and this function is called when receiving it as part
	 * of the shared parameters. These operations are automatically performed by
	 * {@link GNNMiner#newInteractionFromMap(Interaction, String, InteractionType)}
	 * @param neighborAggregation The received Tensor of neighbor aggregation
	 */
	public synchronized void setNeighborAggregation(Tensor neighborAggregation) {
		initializeIfNeeded();
		this.neighborAggregation = neighborAggregation;
	}
	
	/**
	 * An aggregation of the node's neighborhood embeddings in the social graph.
	 * @return A tensor holding the aggregated neighborhood embeddings
	 * @see #setNeighborAggregation(Tensor)
	 */
	public synchronized Tensor getNeighborAggregation() {
		initializeIfNeeded();
		return neighborAggregation;
	}
	
	/**
	 * Sets the regularization (default is a zero vector) of the {@link #updateEmbedding(Tensor)} operation.
	 * Setting this to a value other than a zero tensor helps influence the trained embedding to a latent space around
	 * the given value. Hence, this can be used to influence the embeddings of nodes in the contextual ego network
	 * given their embeddings in other devices.
	 * @param regularization The given regularization tensor.
	 * @see #setRegularizationWeight(double)
	 */
	public synchronized void setRegularization(Tensor regularization) {
		initializeIfNeeded();
		this.regularization = regularization;
	}
	
	/**
	 * Sets the learning rate (default is 1) of the {@link #updateEmbedding(Tensor)} operation.
	 * @param learningRate The given regularization weight.
	 * @return <code>this</code> GNNNodeData instance.
	 */
	public GNNNodeData setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

	/**
	 * Sets the regularization weight (default is 0.1) of the {@link #updateEmbedding(Tensor)} operation.
	 * This helps influence the trained embedding to a latent space around the given value
	 * @param regularizationWeight The given regularization weight.
	 * @return <code>this</code> GNNNodeData instance.
	 * @see #setRegularization(Tensor)
	 */
	public synchronized GNNNodeData setRegularizationWeight(double regularizationWeight) {
		this.regularizationWeight = regularizationWeight;
		return this;
	}
	
	/**
	 * Performs the operation <i>embedding += (embedding-regularization)*learningRate*regularizationWeight-derivative*learningRate</i>
	 * that is a regularized gradient descent over a computed derivative, where the area of regularization is constrained towards the point
	 * set by {@link #setRegularization(Tensor)}.
	 * @param derivative The derivative of the embedding.
	 * @see #getEmbedding()
	 * @see #setLearningRate(double)
	 * @see #setRegularizationWeight(double)
	 * @see #setRegularization(Tensor)
	 */
	public synchronized void updateEmbedding(Tensor derivative) {
		//System.out.println(embedding.subtract(regularization).norm());
		embedding = embedding
						.add(regularization.subtract(embedding).selfMultiply(regularizationWeight*learningRate))
						.selfAdd(derivative.multiply(-learningRate));
						//.setToNormalized();
	}
};
