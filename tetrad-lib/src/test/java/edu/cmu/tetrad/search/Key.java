package edu.cmu.tetrad.search;

public class Key {

	public final int n_a;
	public final int n_d;

	public Key(final int n_a, final int n_d) {
		this.n_a = n_a;
		this.n_d = n_d;
	}
	@Override
	public boolean equals (final Object O) {
		if (!(O instanceof Key)) return false;
		if (((Key) O).n_a != n_a) return false;
		if (((Key) O).n_d != n_d) return false;
		return true;
	}
	 @Override
	 public int hashCode() {
		 return this.n_a ^ this.n_d;
	 }
	 public String print(Key key){
		return "("+key.n_a +", "+ key.n_d + ")";
	 }

}
