package eu.h2020.helios_social.modules.socialgraphmining.heuristics;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This class provides a {@link SocialGraphMiner} that recommends random interactions among
 * previous ones. It has no predictive accuracy and can at best be used as a baseline.
 * 
 * @author Emmanouil Krasanakis
 */
public class RandomMiner extends SocialGraphMiner {

	public RandomMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteractionParameters(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
	}

	@Override
	public SocialGraphMinerParameters getModelParameterObject(Interaction interaction) {
		return null;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		return Math.random();
	}
}
