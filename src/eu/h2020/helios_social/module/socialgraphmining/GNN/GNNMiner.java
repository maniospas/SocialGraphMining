package eu.h2020.helios_social.module.socialgraphmining.GNN;


import java.util.ArrayList;
import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.GNN.operations.Loss;
import eu.h2020.helios_social.module.socialgraphmining.GNN.operations.Tensor;

/**
 * This class provides an implementation of a {@link SocialGraphMiner} based on
 * a Graph Neural Network (GNN) architecture.
 * 
 * @author Emmanouil Krasanakis
 */
public class GNNMiner extends SocialGraphMiner {
	
	public GNNMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		if(neighborModelParameters==null || interaction.getEdge().getEgo()==null)
			return;
		String[] receivedTensors = neighborModelParameters.split("\\;");
		Edge edge = interaction.getEdge();
		edge.getAlter().getOrCreateInstance(GNNNodeData.class).setRegularization(new Tensor(receivedTensors[0]));
		edge.getAlter().getOrCreateInstance(GNNNodeData.class).setNeighborAggregation(new Tensor(receivedTensors[1]));
		if(interactionType==InteractionType.RECEIVE_REPLY) {
			ContextTrainingExampleData trainingExampleData = edge.getContext().getOrCreateInstance(ContextTrainingExampleData.class);
			trainingExampleData.degrade(0.5);
			//create the positive training example
			trainingExampleData.getTrainingExampleList().add(new TrainingExample(edge.getSrc(), edge.getDst(), 1));
			//create two negative training examples
			if(edge.getContext().getNodes().size()>2) {
					Node negativeNode = edge.getSrc();
					while(negativeNode==edge.getSrc())
						negativeNode = edge.getContext().getNodes().get((int)(Math.random()*edge.getContext().getNodes().size()));
					trainingExampleData.getTrainingExampleList().add(new TrainingExample(edge.getSrc(), negativeNode, 0));
					//trainingExampleData.getTrainingExampleList().add(new TrainingExample(negativeNode, edge.getDst(), 0));
				}
			train(trainingExampleData.getTrainingExampleList());
		}
	}
	
	protected Tensor aggregateNeighborEmbeddings(Context context) {
		Node egoNode = context.getContextualEgoNetwork().getEgo();
		Tensor ret = getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy();
		double totalWeight = 0;
		for(TrainingExample trainingExample : context.getOrCreateInstance(ContextTrainingExampleData.class).getTrainingExampleList()) {
			if(trainingExample.getSrc()==egoNode && trainingExample.getLabel()==1) {
				ret.add(trainingExample.getDst().getOrCreateInstance(GNNNodeData.class).getEmbedding().multiply(trainingExample.getWeight()));
				totalWeight += trainingExample.getWeight();
			}
			if(trainingExample.getDst()==egoNode && trainingExample.getLabel()==1) {
				ret.add(trainingExample.getSrc().getOrCreateInstance(GNNNodeData.class).getEmbedding().multiply(trainingExample.getWeight()));
				totalWeight += trainingExample.getWeight();
			}
		}
		if(totalWeight!=0)
			ret = ret.multiply(1./totalWeight);
		return ret;
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		Node egoNode = interaction==null?getContextualEgoNetwork().getEgo():interaction.getEdge().getContextualEgoNetwork().getEgo();
		Context context = interaction==null?getContextualEgoNetwork().getCurrentContext():interaction.getEdge().getContext();
		return egoNode.getOrCreateInstance(GNNNodeData.class).getEmbedding().toString()+";"
			 + aggregateNeighborEmbeddings(context).toString()+";" ;
	}
	
	protected void train(ArrayList<TrainingExample> trainingExamples) {
		double learningRate = 1;
		double previousLoss = -1;
		for(int epoch=0;epoch<150;epoch++) {
			double loss = trainEpoch(trainingExamples, learningRate);
			//System.out.println("Epoch " +epoch+"\t Loss "+ loss);
			learningRate *= 0.95;
			if(Math.abs(previousLoss-loss)<0.001*loss)
				break;
			previousLoss = loss;
		}
	}
	
	protected double trainEpoch(ArrayList<TrainingExample> trainingExamples, double learningRate) {
		HashMap<Node, Tensor> derivatives = new HashMap<Node, Tensor>();
		HashMap<Node, Double> totalWeights = new HashMap<Node, Double>();
		double loss = 0;
		Tensor zero = getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding().zeroCopy();
		for(TrainingExample trainingExample : trainingExamples) {
			Node u = trainingExample.getSrc();
			Node v = trainingExample.getDst();
			{
				Tensor embedding_u = u.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				totalWeights.put(u, totalWeights.getOrDefault(u, 0.)+trainingExample.getWeight());
				totalWeights.put(v, totalWeights.getOrDefault(v, 0.)+trainingExample.getWeight());
				loss += trainingExample.getWeight()*Loss.crossEntropy(Loss.sigmoid(embedding_u.dot(embedding_v)), trainingExample.getLabel());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), trainingExample.getLabel());
				derivatives.put(u, embedding_v.multiply(crossEntropyDerivative).add(derivatives.getOrDefault(u, zero)));
				derivatives.put(v, embedding_u.multiply(crossEntropyDerivative).add(derivatives.getOrDefault(v, zero)));
			}
			{
				Tensor embedding_u = v.getOrCreateInstance(GNNNodeData.class).getNeighborAggregation();
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				totalWeights.put(v, totalWeights.getOrDefault(v, 0.)+trainingExample.getWeight());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), trainingExample.getLabel());
				derivatives.put(v, embedding_u.multiply(crossEntropyDerivative).add(derivatives.getOrDefault(v, zero)));
			}
			if(trainingExample.getLabel()==1)
			{
				Tensor embedding_u = u.getOrCreateInstance(GNNNodeData.class).getEmbedding();
				Tensor embedding_v = v.getOrCreateInstance(GNNNodeData.class).getNeighborAggregation();
				totalWeights.put(u, totalWeights.getOrDefault(u, 0.)+trainingExample.getWeight());
				double crossEntropyDerivative = trainingExample.getWeight()
						*Loss.crossEntropySigmoidDerivative(embedding_u.dot(embedding_v), 1-trainingExample.getLabel());
				derivatives.put(u, embedding_v.multiply(crossEntropyDerivative).add(derivatives.getOrDefault(u, zero)));
			}
		}
		for(Node u : derivatives.keySet()) 
			u.getOrCreateInstance(GNNNodeData.class).updateEmbedding(derivatives.get(u), learningRate);
		return loss;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		if(destinationNode==null)
			Utils.error(new IllegalArgumentException());
		Tensor egoEmbedding = destinationNode.getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding();
		Tensor destinationEmbedding = destinationNode.getOrCreateInstance(GNNNodeData.class).getEmbedding();
		return Loss.sigmoid(egoEmbedding.dot(destinationEmbedding));
	}
}
