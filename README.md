# Social Graph Mining Module

## Introduction
This module aims to provide dynamic machine learning capabilities to HELIOS users.
In particular, each user (or, more precisely, their HELIOS device) carries a different instance of this module 
and needs to account for its capabilities to facilitate recommendation tasks. 

## Dependencies
This module depends on eu.h2020.helios_social.core.contextualegonetwork 

## API Usage
Here we provide example usage of how this module's Graph Neural Network (GNN) miner can be
integrated in in-memory dummy devices to facilitate recommendation capabilities:
```java
import java.util.HashMap;
import java.util.Map.Entry;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.module.socialgraphmining.GNN.GNNMiner;


public class Device {
	private ContextualEgoNetwork contextualEgoNetwork;
	private SocialGraphMiner miner;
	public Device(String name) {
		contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("experiment_data\\", name, null);
		miner = new GNNMiner(contextualEgoNetwork);
		contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default context"));
	}
	public String getName() {
		return contextualEgoNetwork.getEgo().getId();
	}
	public void send(Device other) {
		other.receive(this, miner.getModelParameters(null));
	}
	protected void receive(Device other, String parameters) {
		Interaction interaction = contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getOrCreateNode(other.getName(), null), contextualEgoNetwork.getEgo())
				.addDetectedInteraction(null);
		miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
		other.receiveAck(this, miner.getModelParameters(interaction));
	}
	protected void receiveAck(Device other, String parameters) {
		Interaction interaction = contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
				.addDetectedInteraction(null);
		miner.newInteraction(interaction, parameters, InteractionType.RECEIVE_REPLY);
	}
	public Node recommendNextInteraction() {
		HashMap<Node, Double> interactionScores = miner.recommendInteractions(contextualEgoNetwork.getCurrentContext());
		double bestScore = 0;
		Node bestNode = null;
		for(Entry<Node, Double> interactionScore : interactionScores.entrySet()) 
			if(interactionScore.getValue() > bestScore) {
				bestScore = interactionScore.getValue();
				bestNode = interactionScore.getKey();
			}
		return bestNode;
	}
}
```
An example simulation that demonstrates the capabilities of the mining module in more details
has been setup in the file *eu.h2020.helios_social.module.socialgraphmining.examples.java*.

## Project Structure
This project contains the following components:

src - The source code files.

doc - Additional documentation files.