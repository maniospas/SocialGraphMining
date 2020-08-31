package eu.h2020.helios_social.modules.socialgraphmining.heuristics;


import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;


/**
 * This class provides a {@link SocialGraphMiner} that re-recommends previous interactions 
 * with alters based on the numbeR of previous interactions.
 * 
 * @author Emmanouil Krasanakis
 */
public class ProbabilityMiner extends SocialGraphMiner {
	public ProbabilityMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteractionFromMap(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
	}

	@Override
	public SocialGraphMinerParameters getModelParametersAsMap(Interaction interaction) {
		return null;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		double val = 0;
		for(Edge edge : context.getEdges()) 
			if(edge.getEgo()!=null && edge.getAlter()==destinationNode) 
				val += edge.getInteractions().size();
		return val;
	}

}
