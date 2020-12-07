package eu.h2020.helios_social.modules.socialgraphmining.diffusion;

/*public class PPRMiner extends SocialGraphMiner {
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
	public double predictNewInteraction(Context context, Node destinationNode) {
		if(destinationNode==null)
			Utils.error(new IllegalArgumentException());
		Tensor egoEmbedding = destinationNode.getContextualEgoNetwork().getEgo().getOrCreateInstance(GNNNodeData.class).getEmbedding();
		Tensor destinationEmbedding = destinationNode.getOrCreateInstance(GNNNodeData.class).getEmbedding();;
		return Loss.sigmoid(egoEmbedding.dot(destinationEmbedding));
	}

	@Override
	public void newInteractionFromMap(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
		
	}

	@Override
	public SocialGraphMinerParameters getModelParametersAsMap(Interaction interaction) {
		return null;
	}

}*/
