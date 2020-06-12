package eu.h2020.helios_social.modules.socialgraphmining.heuristics;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;


/**
 * This class provides a {@link SocialGraphMiner} that re-recommends previous interactions 
 * with alters based on their chronological order.
 * 
 * @author Emmanouil Krasanakis
 */
public class RepeatMiner extends SocialGraphMiner {
	/**
	 * This class is used to hold information about the order alters have been interacted with.
	 * It is tied to the edge of each interaction.
	 * 
	 * @author Emmanouil Krasanakis
	 */
	public static class SendOrderTimestamp {
		private static long maxValue = 0;
		private long value = 0;
		public SendOrderTimestamp() {}
		public void updateValue() {
			maxValue += 1;
			value = maxValue;
		}
		public long getValue() {
			//if(value==0)
			//	Utils.error("RepeatMiner have not assigned an OrderTimestamp to the edge");
			return value;
		}
	}
	
	public RepeatMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		if(interactionType==InteractionType.RECEIVE_REPLY) 
			interaction.getEdge().getOrCreateInstance(SendOrderTimestamp.class).updateValue();
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		return null;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		for(Edge edge : context.getEdges())
			if(edge.getAlter()==destinationNode)
				return edge.getOrCreateInstance(SendOrderTimestamp.class).getValue();
		return 0;
	}

}
