# Social Graph Mining Module

## Introduction
This module aims to provide dynamic machine learning capabilities to HELIOS users.
In particular, each user (or, more precisely, their HELIOS device) carries a different instance of this module 
and needs to account for its capabilities to facilitate recommendation tasks. 

## API Usage
Here we provide example usage of this module's Graph Neural Network (GNN) miner to facilitate recommendation capabilities:
```java
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.GNNMiner;

/// retrieval of the contextual ego network
ContextualEgoNetwork cen = ...;

// when starting the miner, use the contextual ego network as reference
SocialGraphMiner miner = new GNNMiner(cen);

// when sending a message, retrieve its interaction, pass it to the miner and send model parameters
Interaction interaction = ...;
miner.newInteraction(interaction);
String parametersToSend = miner.getModelParameters(interaction);

// when receiving a message, retrieve the parameters and interaction created in the contextual ego network and pass them to the miner
Interaction interaction = ...;
String receirvedParameters = ...;
miner.newInteraction(interaction, receirvedParameters, false);

// to recommend plausible edges in which to form new interactions call
miner.predictOutgoingInteractions();
```


## Project Structure
This project contains the following components:

src - The source code files.

doc - Additional documentation files.