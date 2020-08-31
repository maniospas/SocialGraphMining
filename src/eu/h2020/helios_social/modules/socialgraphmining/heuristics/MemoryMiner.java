package eu.h2020.helios_social.modules.socialgraphmining.heuristics;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

public class MemoryMiner extends SocialGraphMiner {
	private SocialGraphMiner baseMiner;
	private HashMap<Node, Double> smoothen = new HashMap<Node, Double>();

	public MemoryMiner(SocialGraphMiner baseMiner) {
		super(baseMiner.getContextualEgoNetwork());
		this.baseMiner = baseMiner;
	}

	@Override
	public void newInteractionFromMap(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
		baseMiner.newInteractionFromMap(interaction, neighborModelParameters, interactionType);
	}

	@Override
	public SocialGraphMinerParameters getModelParametersAsMap(Interaction interaction) {
		return baseMiner.getModelParametersAsMap(interaction);
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		double previous = smoothen.getOrDefault(destinationNode, 0.);
		double prediction = baseMiner.predictNewInteraction(context, destinationNode);
		double value = 0.5*previous + prediction*0.5;
		smoothen.put(destinationNode, value);
		return value;
	}

}
