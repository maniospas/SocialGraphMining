package eu.h2020.helios_social.module.socialgraphmining.heuristics;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;


/**
 * This class provides a {@link SocialGraphMiner} that re-recommends previous interactions 
 * with alters based on the order of received messages.
 * 
 * @author Emmanouil Krasanakis
 */
public class ReplyMiner extends SocialGraphMiner {
	/**
	 * This class is used to hold information about the order alters have been interacted with.
	 * It is tied to the edge of each interaction.
	 * 
	 * @author Emmanouil Krasanakis
	 */
	public static class ReceiveOrderTimestamp {
		private static long maxValue = 0;
		private long value = 0;
		public ReceiveOrderTimestamp() {}
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
	
	public ReplyMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		if(interactionType==InteractionType.RECEIVE) 
			interaction.getEdge().getOrCreateInstance(ReceiveOrderTimestamp.class).updateValue();
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		return null;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		for(Edge edge : context.getEdges())
			if(edge.getAlter()==destinationNode)
				return edge.getOrCreateInstance(ReceiveOrderTimestamp.class).getValue();
		return 0;
	}

}
