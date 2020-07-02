package eu.h2020.helios_social.modules.socialgraphmining;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;

/**
 * Provides an abstraction of the basic capabilities and requirements of graph mining algorithms.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class SocialGraphMiner {
	public enum InteractionType {SEND, RECEIVE, RECEIVE_REPLY};
	
	private ContextualEgoNetwork contextualEgoNetwork;
	protected SocialGraphMiner(ContextualEgoNetwork contextualEgoNetwork) {
		if(contextualEgoNetwork==null)
			throw new IllegalArgumentException();
		this.contextualEgoNetwork = contextualEgoNetwork;
	}
	public ContextualEgoNetwork getContextualEgoNetwork() {
		return contextualEgoNetwork;
	}
	/** 
	 * Makes the graph miner aware that a user received an interaction from another user with {@link #getModelParameters}.
	 * @param interaction A new interaction the user initiates expressed in terms of the contextual ego network
	 * @param neighborModelParameters The neighbor parameters. May be null for when interactionType==SEND.
	 * @param interactionType The type of the interaction (SEND, RECEIVE or RECEIVE_REPLY corresponds to acknowledging the receive).
	 */
	public abstract void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType);
    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A String serialization of model parameters.
     */
    public abstract String getModelParameters(Interaction interaction);
    /**
     * Predicts the weight of performing a SEND interaction between the given context's ego and a destination node
     * within a given context. This method should typically return values in the [0,1] range, where higher values 
     * correspond to more likely interactions.
     * @param context The context in which to perform the prediction.
     * @param destinationNode The destination node of the interaction.
     * @return The weight of performing the interaction (higher values are more likely).
     */
    public abstract double predictNewInteraction(Context context, Node destinationNode);
    /**
     * Calls {@link #predictNewInteraction(Context, Node)} to score the likelihood of interacting with all nodes
     * of the given context.
     * @param context The context for which to recommend interactions.
     * @return A hash map of node scores (larger is more likely to occur.
     */
    public HashMap<Node, Double> recommendInteractions(Context context) {
    	HashMap<Node, Double> scores = new HashMap<Node, Double>();
    	Node ego = context.getContextualEgoNetwork().getEgo();
    	for(Node node : context.getNodes())
    		if(node!=ego)
    			scores.put(node, predictNewInteraction(context, node));
    	return scores;
    }
}