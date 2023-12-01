use std::ops::Add;
use std::ops::Mul;

pub mod Value{
	struct Value{
		data_: f64,
		gradient_: f64,
		prev_: (&Value, &Value),
	}
	
	impl Value{
		fn new(data: f64) -> Value{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: (),
			}
		}
	}
	
	impl ToString for Value{
		fn to_string(&self) -> String{
			format!("Value(data:{}, gradient:{})", self.data_, self.gradient_)
		}
	}
	
	impl Add for Value{
		fn add(&self, other: &Value) -> Value{
			Value{
				data_: self.data_ + other.data_,
				gradient_: 0.0,
				prev_:(self, other),
			}
		}
	}
	
	impl Mul for Value{
		fn mul(&self, other: &Value) -> Value{
			Value{
				data_: self.data_ * other.data_,
				gradient_: 0.0,
				prev_:(self, other),
			}
		}
	}
}
