package eu.h2020.helios_social.modules.socialgraphmining;

import java.util.HashMap;
import java.util.Set;

public class SocialGraphMinerParameters {
	private HashMap<String, Object> params;
	public SocialGraphMinerParameters() {
		params = new HashMap<String, Object>();
	}
	public Object get(String key) {
		return params.get(key);
	}
	public SocialGraphMinerParameters getNested(String key) {
		return (SocialGraphMinerParameters)get(key);
	}
	public void put(String key, Object value) {
		params.put(key, value);
	}
	public Set<String> getKeys() {
		return params.keySet();
	}
}
