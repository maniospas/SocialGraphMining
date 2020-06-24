package eu.h2020.helios_social.modules.socialgraphmining.GNN;


import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Loss;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;

/**
 * This class provides an implementation of a {@link SocialGraphMiner} based on
 * a Graph Neural Network (GNN) architecture. Its parameters can be adjusted using a number of
 * setter methods.
 * 
 * @author Emmanouil Krasanakis
 */
public class GNNMiner extends SocialGraphMiner {
	private double learningRate = 1;
	private double learningRateDegradation = 0.95;
	private double regularizationWeight = 0.1;
	private double regularizationAbsorbsion = 1;
	private int maxEpoch = 1000;
	private double convergenceRelativeLoss = 0.001;
	private double trainingExampleDegradation = 0.5;
	private double trainingExampleRemovalThreshold = 0.01;
	private double egoDeniability = 0;
	private double neighborDeniability = 0;
	private double incommingEdgeLearningRateMultiplier = 0;
	private double outgoingEdgeLearningRateMultiplier = 0;
	private double updateEgoEmbeddingsFromNeighbors = 0;
	
	/**
	 * Instantiates a {@link GNNMiner} on a given contextual ego network.
	 * @param contextualEgoNetwork The contextual ego network on which the miner runs and stores information.
	 */
	public GNNMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}
	
	/***
	 * The learning rate (default is 1) from which GNNMiner training starts. Training restarts
	 * on each {@link #newInteraction} from this value and can be potentially be adapted by
	 * {@link #setLearningRateDegradation} over training epochs.
	 * @param learningRate The learning rate to restart from.
	 * @return <code>this</code> GNNMiner instance.
	 * @see GNNNodeData#setLearningRate(double)
	 */
	public GNNMiner setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}
	
	public GNNMiner setEdgePointsLearningMultiplier(double incomming, double outgoing) {
		incommingEdgeLearningRateMultiplier = incomming;
		outgoingEdgeLearningRateMultiplier = outgoing;
		return this;
	}

	/**
	 * Performs a fixed degradation of the learning rate over training epochs by multiplying the latter
	 * with a given factor (default is 0.95) after each epoch.
	 * @param learningRateDegradation The rate at which learning rate degrades.
	 * @return <code>this</code> GNNMiner instance.
	 * @see #setLearningRate(double)
	 */
	public GNNMiner setLearningRateDegradation(double learningRateDegradation) {
		this.learningRateDegradation = learningRateDegradation;
		return this;
	}
	
	/**
	 * The regularization weight (default 0.1) to apply during training of the GNNMiner.
	 * This weight ensures that training converges around given areas of the embedding space.
	 * @param regularizationWeight The regularization weight to set.
	 * @return <code>this</code> GNNMiner instance.
	 * @see GNNNodeData#setRegularizationWeight(double)
	 * @see #setRegularizationAbsorbsion(double)
	 */
	public GNNMiner setRegularizationWeight(double regularizationWeight) {
		this.regularizationWeight = regularizationWeight;
		return this;
	}
	

	/**
	 * Multiplies regularization tensors with this value before setting them as regularization;
	 * value of 1 (default) produces regularization of calculated alter embeddings towards the
	 * embeddings calculated on alter devices. Value of 0 produce regularization towards zero,
	 * which effectively limits the embedding norm to the approximate order of magnitude
	 * 1/weight, where weight is the value set to {@link #setRegularizationWeight(double)}.
	 * @param absorbsion
	 * @return <code>this</code> GNNMiner instance.
	 */
	public GNNMiner setRegularizationAbsorbsion(double regularizationAbsorbsion) {
		this.regularizationAbsorbsion = regularizationAbsorbsion;
		return this;
	}
	
	/**
	 * Limits the number of training epochs (default is 1000) over which to
	 * train the GNNMiner.
	 * @param maxEpoch The maximum epoch at which to train.
	 * @return <code>this</code> GNNMiner instance.
	 */
	public GNNMiner setMaxTrainingEpoch(int maxEpoch) {
		this.maxEpoch = maxEpoch;
		return this;
	}
	
	/**
	 * When the GNNMiner is being trained, training stops at epochs where
	 * abs(previous epoch loss - this epoch loss) < convergenceRelativeLoss*(this epoch loss)
	 * where losses are weighted cross entropy ones. Default is 0.001.
	 * @param convergenceRelativeLoss The relative loss at which to stop training.
	 * @return <code>this</code> GNNMiner instance.
	 */
	public GNNMiner setMinTrainingRelativeLoss(double convergenceRelativeLoss) {
		this.convergenceRelativeLoss = convergenceRelativeLoss;
		return this;
	}
	
	/**
	 * Degrades example weights each time a new one is generated through {@link #newInteraction} by calling
	 * {@link ContextTrainingExampleData#degrade} to multiply previous weights with the given degradation factor
	 * (default is 0.5).
	 * @param trainingExampleDegradation The factor with which to multiply each 8previous example weight
	 * @return <code>this</code> GNNMiner instance.
	 * @see #setTrainingExampleRemovalThreshold(double)
	 */
	public GNNMiner setTrainingExampleDegradation(double trainingExampleDegradation) {
		this.trainingExampleDegradation = trainingExampleDegradation;
		return this;
	}
	
	/**
	 * Sets the threshold weight at which old training examples are removed (default is 0.01).
	 * Basically, if the degradation set by set by {@link #setTrainingExampleDegradation} 
	 * remains constant throughout training iterations, training examples are removed if
	 * degradation^n < trainingExampleRemovalThreshold
	 * where n the number of (positive) examples provided after the examined one with {@link #newInteraction}.
	 * 
	 * @param trainingExampleRemovalThreshold The weight threshold at which to remove GNN training examples which is passed as
	 * 	the second argument to {@link ContextTrainingExampleData#degrade} calls.
	 * @return <code>this</code> GNNMiner instance.
	 */
	public GNNMiner setTrainingExampleRemovalThreshold(double trainingExampleRemovalThreshold) {
		this.trainingExampleRemovalThreshold = trainingExampleRemovalThreshold;
		return this;
	}
	
	/**
	 * Enables plausible deniability and differential privacy handling by permuting the ego and its alter's parameters
	 * with a random noise proportional to a given constant and their norm. Zero values (default) ensure no privacy concerns
	 * but more exact  computations <i>for other</i> devices. The user's device would perform predictions depending on the
	 * privacy settings of their alters.
	 * 
	 * @param plausibleDeniability The permutation of the ego's parameters.
	 * @param differentialPrivacy The permutation of the neighbor's parameters.
	 * @return <code>this</code> GNNMiner instance.
	 */
	public GNNMiner setDeniability(double plausibleDeniability, double differentialPrivacy) {
		this.egoDeniability = plausibleDeniability;
		this.neighborDeniability = differentialPrivacy;
		return this;
	}

	@Override
	public synchronized void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		if(neighborModelParameters==null || interaction.getEdge().getEgo()==null || interactionType==InteractionType.SEND)
			return;
		String[] receivedTensors = neighborModelParameters.split("\\;");
		Edge edge = interaction.getEdge();
		if(updateEgoEmbeddingsFromNeighbors!=0)
			edge.getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().selfMultiply(1-updateEgoEmbeddingsFromNeighbors).selfAdd(new Tensor(receivedTensors[2]).selfMultiply(updateEgoEmbeddingsFromNeighbors));
		edge.getAlter().getOrCreateInstance(GNNNodeData.class).setRegularization((new Tensor(receivedTensors[0])).selfMultiply(regularizationAbsorbsion));
		edge.getAlter().getOrCreateInstance(GNNNodeData.class).setNeighborAggregation(new Tensor(receivedTensors[1]));
		if(interactionType==InteractionType.RECEIVE_REPLY || interactionType==InteractionType.RECEIVE) {
			ContextTrainingExampleData trainingExampleData = edge.getContext().getOrCreateInstance(ContextTrainingExampleData.class);

			if(trainingExampleData.transformToDstEmbedding==null) 
				trainingExampleData.transformToDstEmbedding = edge.getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy().setToOnes();
			if(trainingExampleData.transformToSrcEmbedding==null) 
				trainingExampleData.transformToSrcEmbedding = edge.getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy().setToOnes();
			
			if(trainingExampleDegradation!=1)
				trainingExampleData.degrade(trainingExampleDegradation, trainingExampleRemovalThreshold);
			//create the positive training example
			trainingExampleData.getTrainingExampleList().add(new TrainingExample(edge.getSrc(), edge.getDst(), 1));
			//create two negative training examples
			if(edge.getContext().getNodes().size()>2) {
					Node negativeNode = edge.getSrc();
					while(negativeNode==edge.getSrc() || negativeNode==edge.getDst())
						negativeNode = edge.getContext().getNodes().get((int)(Math.random()*edge.getContext().getNodes().size()));
					trainingExampleData.getTrainingExampleList().add(new TrainingExample(edge.getSrc(), negativeNode, 0));
					trainingExampleData.getTrainingExampleList().add(new TrainingExample(negativeNode, edge.getDst(), 0));
				}
			train(trainingExampleData);
		}
	}
	
	protected Tensor aggregateNeighborEmbeddings(Context context) {
		Node egoNode = context.getContextualEgoNetwork().getEgo();
		Tensor ret = getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy();
		double totalWeight = 0;
		for(TrainingExample trainingExample : context.getOrCreateInstance(ContextTrainingExampleData.class).getTrainingExampleList()) {
			if(trainingExample.getSrc()==egoNode) {
				ret.selfAdd(trainingExample.getDst().getOrCreateInstance(GNNNodeData.class)
						.getEmbedding()
						.multiply(trainingExample.getWeight()*(trainingExample.getLabel()-0.5))
						);
				/*ret.selfAdd(trainingExample.getDst().getOrCreateInstance(GNNNodeData.class)
						.getNeighborAggregation()
						.multiply(-trainingExample.getWeight()*(trainingExample.getLabel()-0.5))
						);*/
				totalWeight += trainingExample.getWeight();
			}
			if(trainingExample.getDst()==egoNode) {
				ret.selfAdd(trainingExample.getSrc().getOrCreateInstance(GNNNodeData.class)
						.getEmbedding()
						.multiply(trainingExample.getWeight())
						.multiply(trainingExample.getWeight()*(trainingExample.getLabel()-0.5))
						);
				/*ret.selfAdd(trainingExample.getSrc().getOrCreateInstance(GNNNodeData.class)
						.getNeighborAggregation()
						.multiply(trainingExample.getWeight())
						.multiply(-trainingExample.getWeight()*(trainingExample.getLabel()-0.5))
						);*/
				totalWeight += trainingExample.getWeight();
			}
		}
		if(totalWeight!=0)
			ret = ret.multiply(1./totalWeight);
		return ret;
	}
	
	protected Tensor permute(Tensor tensor, double permutation) {
		if(permutation==0)
			return tensor;
		return tensor
				.multiply(1-permutation)
				.selfAdd(tensor.zeroCopy().setToRandom().multiply(tensor.norm()*permutation));
	}
	
	@Override
	public String getModelParameters(Interaction interaction) {
		if(interaction==null) 
			return Utils.error("Could not find given context", null);
		Context context = interaction.getEdge().getContext();
		return permute(interaction.getEdge().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding(), egoDeniability).toString()+";"
			 + permute(aggregateNeighborEmbeddings(context), neighborDeniability).toString()+";"
			 + permute(interaction.getEdge().getAlter().getOrCreateInstance(GNNNodeData.class).getEmbedding(), neighborDeniability).toString()+";";
	}
	
	protected void train(ContextTrainingExampleData trainingExampleData) {
		double learningRate = this.learningRate;
		double previousLoss = -1;
		for(int epoch=0;epoch<maxEpoch;epoch++) {
			double loss = trainEpoch(trainingExampleData, learningRate);
			//System.out.println("Epoch " +epoch+"\t Loss "+ loss);
			learningRate *= this.learningRateDegradation;
			if(Math.abs(previousLoss-loss)<convergenceRelativeLoss*loss)
				break;
			previousLoss = loss;
		}
	}
	
	protected double trainEpoch(ContextTrainingExampleData trainingExampleData, double learningRate) {
		Tensor zero = getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy();
		HashMap<Node, Tensor> derivatives = new HashMap<Node, Tensor>();
		HashMap<Node, Double> totalWeights = new HashMap<Node, Double>();
		Tensor transformToSrcEmbeddingDerivative = zero.zeroCopy();
		Tensor transformToDstEmbeddingDerivative = zero.zeroCopy();
		double transformToSrcEmbeddingDerivativeWeight = 0;
		double transformToDstEmbeddingDerivativeWeight = 0;
		double loss = 0;
		for(TrainingExample trainingExample : trainingExampleData.getTrainingExampleList()) {
			Node u = trainingExample.getSrc();
			Node v = trainingExample.getDst();
			{
				Tensor embedding_u = u.getOrCreateInstance(GNNNodeData.class).getEmbedding().multiply(trainingExampleData.transformToSrcEmbedding);
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getEmbedding().multiply(trainingExampleData.transformToDstEmbedding);
				totalWeights.put(u, totalWeights.getOrDefault(u, 0.)+trainingExample.getWeight());
				totalWeights.put(v, totalWeights.getOrDefault(v, 0.)+trainingExample.getWeight());
				transformToSrcEmbeddingDerivativeWeight += trainingExample.getWeight();
				transformToDstEmbeddingDerivativeWeight += trainingExample.getWeight();
				loss += trainingExample.getWeight()*Loss.crossEntropy(Loss.sigmoid(embedding_u.dot(embedding_v)), trainingExample.getLabel());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), trainingExample.getLabel());
				derivatives.put(u, embedding_v
									.multiply(trainingExampleData.transformToSrcEmbedding)
									.selfMultiply(crossEntropyDerivative)
									.selfAdd(derivatives.getOrDefault(u, zero)));
				derivatives.put(v, embedding_u
									.multiply(trainingExampleData.transformToDstEmbedding)
									.selfMultiply(crossEntropyDerivative)
									.selfAdd(derivatives.getOrDefault(v, zero)));
				/*transformToSrcEmbeddingDerivative = u.getOrCreateInstance(GNNNodeData.class).getEmbedding()
									.multiply(embedding_v)
									.selfMultiply(crossEntropyDerivative)
									.selfAdd(transformToSrcEmbeddingDerivative);
				transformToDstEmbeddingDerivative = v.getOrCreateInstance(GNNNodeData.class).getEmbedding()
									.multiply(embedding_u)
									.selfMultiply(crossEntropyDerivative)
									.selfAdd(transformToDstEmbeddingDerivative);*/
			}
			/*if(trainingExample.getLabel()==1)
			{
				Tensor embedding_u = v.getOrCreateInstance(GNNNodeData.class).getNeighborAggregation();
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				totalWeights.put(v, totalWeights.getOrDefault(v, 0.)+trainingExample.getWeight());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), 1);
				derivatives.put(v, embedding_u.multiply(crossEntropyDerivative).add(derivatives.getOrDefault(v, zero)));
			}
			if(trainingExample.getLabel()==1)
			{
				Tensor embedding_u = u.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getNeighborAggregation();
				totalWeights.put(u, totalWeights.getOrDefault(u, 0.)+trainingExample.getWeight());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), 1);
				derivatives.put(u, embedding_v.multiply(crossEntropyDerivative).selfAdd(derivatives.getOrDefault(u, zero)));
			}*/
		}
		for(Node u : derivatives.keySet()) {
			u.getOrCreateInstance(GNNNodeData.class)
				.setLearningRate(learningRate)
				.setRegularizationWeight(regularizationWeight)
				.updateEmbedding(derivatives.get(u).multiply(1./totalWeights.get(u)));
		}
		System.exit(1);
		
		if(outgoingEdgeLearningRateMultiplier!=0)
			trainingExampleData.transformToSrcEmbedding = transformToSrcEmbeddingDerivative
				.multiply(-outgoingEdgeLearningRateMultiplier*learningRate/transformToSrcEmbeddingDerivativeWeight)
				.add(trainingExampleData.transformToSrcEmbedding.selfMultiply(1-regularizationWeight*outgoingEdgeLearningRateMultiplier));
		if(incommingEdgeLearningRateMultiplier!=0)
			trainingExampleData.transformToDstEmbedding = transformToDstEmbeddingDerivative
					.multiply(-incommingEdgeLearningRateMultiplier*learningRate/transformToDstEmbeddingDerivativeWeight)
					.add(trainingExampleData.transformToDstEmbedding.selfMultiply(1-regularizationWeight*incommingEdgeLearningRateMultiplier));
		return loss;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		if(destinationNode==null)
			Utils.error(new IllegalArgumentException());
		Tensor egoEmbedding = destinationNode.getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding();
		Tensor destinationEmbedding = destinationNode.getOrCreateInstance(GNNNodeData.class).getEmbedding();
		ContextTrainingExampleData trainingExampleData = context.getOrCreateInstance(ContextTrainingExampleData.class);
		egoEmbedding = egoEmbedding.multiply(trainingExampleData.transformToSrcEmbedding);
		destinationEmbedding = destinationEmbedding.multiply(trainingExampleData.transformToDstEmbedding);
		return Loss.sigmoid(egoEmbedding.dot(destinationEmbedding));
	}
}
