
use finalProject::Value;
 
fn main() {
    println!("Hello, world!");
    let mut a = Value::new(6.0);
	let mut b = Value::new(5.0);
	let mut d = Value::new(7.0);
	let mut a1 = Value::new(0.02);
	let mut c = Value::mul(a, b);
	let mut e = Value::add(d, c);
	let mut f = Value::mul(e, a1);
	let mut g = Value::tanh(f);
	g.backwards();
	print_vals(&g);
	//print!("e:{}", e.to_string());
}

fn print_vals(v: &Value){

	if v.prev_.len() == 1{
		print!("tanh()");
		print_vals(&v.prev_.get(0).unwrap());
		println!();
	}
	else if v.prev_.len() == 2{
		print!("  EQ-");
		print_vals(&v.prev_.get(0).unwrap());
		print!("{}", v.op_);
		print_vals(&v.prev_.get(1).unwrap());
		print!("-EQ  ");
		println!();
	}

	print!("{}", v.to_string());
}
