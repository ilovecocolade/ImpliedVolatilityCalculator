CODE NOTES FOR IMPLIED VOLATILITY CALCULATOR - AUTHOR: JOHN ALLEN - johnnyallen@blueyonder.co.uk

1. Place input.csv file in the same directory as imc.py
2. Run imc.py through terminal using >python imc.py
3. Progress will appear in terminal along with any issues, output.csv will be delivered to same directory as imc.py

 - Unit testing file is test_imc.py
 This file test all functionality of Trade class except volatility calculations as the best method of calculating IV was used 
in the application and therefore no validation on the results can be made. 

In order to change the method of calculations replace one of the following functions into the calculate_volatility() method in Trade class:

1. newton_raphson() - this method uses NR method to determine IV for Black-Scholes and Bachelier options.

2. conversion_newton_raphson() - this method uses NR to determine IV for Black_Scholes options then converts from 
			         lognormal to normal volatility for Bachelier options.

3. secant() - this method uses Secant method to determine IV for Black-Scholes and Bachelier options.

4. conversion_secant() - this method uses Secant to determine IV for Black_Scholes options then converts from 
			 lognormal to normal volatility for Bachelier options.

5. false_position() - this method uses False Position method to determine IV for Black-Scholes and Bachelier options.
		      This method also guarantees convergence at the cost of computation speed and no. of operations required.

6. conversion_false_position() - this method uses False Position to determine IV for Black_Scholes options then converts from 
			         lognormal to normal volatility for Bachelier options.


conversion_newton_raphson() was used in the final application because of issues with speed and convergence with Bachelier Options
			    using the NR method without conversion. NR still somethimes fails to converge as vega value is 0. 
			    Use false position method for guaranteed convergence at cost of speed.