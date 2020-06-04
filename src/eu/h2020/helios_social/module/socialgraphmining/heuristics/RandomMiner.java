package eu.h2020.helios_social.module.socialgraphmining.heuristics;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;

/**
 * This class provides a {@link SocialGraphMiner} that recommends random interactions among
 * previous ones. It has not predictive accuracy and should only be used as a baseline.
 * 
 * @author Emmanouil Krasanakis
 */
public class RandomMiner extends SocialGraphMiner {

	public RandomMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		return null;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		return Math.random();
	}
}
