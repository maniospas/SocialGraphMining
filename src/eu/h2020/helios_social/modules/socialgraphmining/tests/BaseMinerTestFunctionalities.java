package eu.h2020.helios_social.modules.socialgraphmining.tests;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.Collectors;

import org.junit.Before;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Serializer;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;

public class BaseMinerTestFunctionalities {
	private HashMap<String, TestDevice> testDevices;
	
	public static String argmax(HashMap<Node, Double> recommendations) {
		return argmax(recommendations, 0);
	}
	public static String argmax(HashMap<Node, Double> recommendations, int position) {
		@SuppressWarnings({ "unchecked", "rawtypes" })
		ArrayList<Node> list = new ArrayList<Node>(recommendations
				.entrySet()
				.stream()
			    .sorted((e1, e2) -> -((Comparable) e1.getValue()).compareTo(e2.getValue()))
			    .map(e -> e.getKey())
			    .collect(Collectors.toList())
		);
		return list.get(position).getId();
	}
	
	@Before
	public void initializeTest() {
		Utils.development = false;
		Serializer.clearSerializers();
		testDevices = new HashMap<String, TestDevice>();
	}
	
	public TestDevice getDevice(String name) {
		TestDevice device = testDevices.get(name);
		if(device==null)
			testDevices.put(name, device = new TestDevice(name));
		return device;
	}

}
