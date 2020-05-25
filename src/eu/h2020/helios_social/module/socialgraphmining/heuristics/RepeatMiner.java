package eu.h2020.helios_social.module.socialgraphmining.heuristics;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;

public class RepeatMiner extends SocialGraphMiner {

	protected RepeatMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		return "";
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		Edge edge = context.getEdge(context.getContextualEgoNetwork().getEgo(), destinationNode);
		if(edge==null || edge.getInteractions().isEmpty())
			return 0;
		return edge.getInteractions().get(edge.getInteractions().size()-1).getStartTime();
	}
	
}
