package eu.h2020.helios_social.module.socialgraphmining;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;

/**
 * Provides an abstraction of the basic capabilities and requirements of mining module algorithms.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class SocialGraphMiner {
	public enum InteractionType {SEND, RECEIVE, RECEIVE_REPLY};
	
	private ContextualEgoNetwork contextualEgoNetwork;
	protected SocialGraphMiner(ContextualEgoNetwork contextualEgoNetwork) {
		this.contextualEgoNetwork = contextualEgoNetwork;
	}
	public ContextualEgoNetwork getContextualEgoNetwork() {
		return contextualEgoNetwork;
	}
	/** 
	 * Makes the graph miner aware that a user received an interaction from another user with {@link #getModelParameters}.
	 * @param interaction A new interaction the user initiates expressed in terms of the contextual ego network
	 * @param neighborModelParameters The neighbor parameters. May be null for when interactionType==SEND.
	 * @param interactionType The type of interaction
	 */
	public abstract void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType);
    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A String serialization of model parameters.
     */
    public abstract String getModelParameters(Interaction interaction);
    /**
     * Predicts the weight of outgoing interactions
     * @return A map of edge weights.
     */
    //public abstract Map<Edge, Double> predictOutgoingInteractions();
    /**
     * Provides various measures concerning the efficacy of predicting the given interaction using
     * the {@link #predictOutgoingInteractions()} method.
     * Needs be called before this interaction is registered (e.g. with {@link #newInteraction}.
     * Some of these measures assume binary (0/1) values and need be averaged across many interactions
     * to provide meaningful insights.
     * @param interaction The interaction to evaluate, expressed in terms of the contextual ego network
     * @return A hashmap between measures and values.
     */
    //public abstract Map<String, Double> evaluate(Interaction interaction);
}