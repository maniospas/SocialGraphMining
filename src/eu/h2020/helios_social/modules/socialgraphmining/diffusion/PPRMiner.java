package eu.h2020.helios_social.modules.socialgraphmining.diffusion;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNNodeData;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Loss;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;

public class PPRMiner extends SocialGraphMiner {
	private double alpha = 0.85;
	
	public PPRMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
		GNNNodeData egoData = contextualEgoNetwork.getEgo().getOrCreateInstance(GNNNodeData.class);
		egoData.setRegularization(egoData.getEmbedding().zeroCopy().setToRandom());
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		if(interactionType==InteractionType.SEND || interactionType==InteractionType.RECEIVE) {
			Tensor neighborParameters = new Tensor(neighborModelParameters);
			interaction.getEdge().getAlter().getOrCreateInstance(GNNNodeData.class)
				.getEmbedding().setToZero()
				.selfAdd(neighborParameters);
			GNNNodeData egoData = interaction.getEdge().getEgo().getOrCreateInstance(GNNNodeData.class);
			egoData
				.setRegularizationWeight(1)
				.setLearningRate(1)
				.updateEmbedding(neighborParameters.selfMultiply(1-alpha));
		}
		
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		Context context = interaction==null?getContextualEgoNetwork().getCurrentContext():interaction.getEdge().getContext();
		if(context==null) 
			return Utils.error("Could not find given context", null);
		Node egoNode = context.getContextualEgoNetwork().getEgo();
		return egoNode.getOrCreateInstance(GNNNodeData.class).getEmbedding().toString();
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		if(destinationNode==null)
			Utils.error(new IllegalArgumentException());
		Tensor egoEmbedding = destinationNode.getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding();
		Tensor destinationEmbedding = destinationNode.getOrCreateInstance(GNNNodeData.class).getEmbedding();;
		return Loss.sigmoid(egoEmbedding.dot(destinationEmbedding));
	}

}
