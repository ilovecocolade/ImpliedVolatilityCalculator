# IMPLIED VOLATILITY CALCULATOR
# AUTHOR - JOHN ALLEN - johnnyallen@blueyonder.co.uk

# BLACK-SCHOLES IMPLIED VOLATILITY CAN BE CALCULATED USING THE NEWTON-RAPHSON METHOD, SECANT METHOD OR FALSE POSITION METHOD
# BACHELIER (NORMAL) IMPLIED VOLATILITY CAN BE CALCULATED USING THE NEWTON-RAPHSON METHOD, SECANT METHOD OR FALSE POSITION METHOD

import pandas as pd  # pandas is used to read in and output csv files as well as build output dataframe from calculated parameters
from math import exp, log, sqrt, nan, pi
from scipy.stats import norm  # cumulative normal distribution function used in Black-Scholes
import decimal
import time
import datetime as dt


# Class to encapsulate each trade for the input data
class Trade:

    # initialise trade object & extract parameters as attributes
    def __init__(self, trade_data):

        self.id = trade_data['ID']
        self.underlying_type = trade_data['Underlying Type']
        self.underlying = trade_data['Underlying']
        self.risk_free_rate = trade_data['Risk-Free Rate']
        self.days_to_expiry = trade_data['Days To Expiry']
        self.strike = trade_data['Strike']
        self.option_type = trade_data['Option Type']
        self.model_type = trade_data['Model Type']
        self.market_price = trade_data['Market Price']

    # method to get trade ID
    def get_id(self):
        if not isinstance(self.id, int):
            raise TypeError('Trade ID must be a positive integer')
        elif self.id < 0:  # if given ID is not a positive integer raise error
            raise ValueError('Trade ID must be positive')
        else:
            return self.id

    # method to get underlying price
    def get_underlying(self):  # if underlying is not a umber greater than 0 raise error
        if not isinstance(self.underlying, int) and not isinstance(self.underlying, float):
            raise TypeError('Underlying Price must be a number greater than 0')
        elif self.underlying <= 0:
            raise ValueError('Underlying Price must be greater than 0')
        else:
            return self.underlying

    # method to get spot price
    def get_spot(self):
        if not isinstance(self.underlying_type, str):  # if a non-valid type is presented
            raise TypeError('Underlying Type must be -Stock- nor -Future-')

        elif self.underlying_type == 'Future':
            return self.get_underlying()  # if Option is a Future return spot price as value of underlying

        elif self.underlying_type == 'Stock':
            return self.get_market_price()  # if Option is a Stock return spot price as current market price

        else:  # if a non-valid entry is presented
            raise ValueError('Underlying Type must be -Stock- nor -Future-')

    # method to get strike price
    def get_strike(self):  # if input strike is not a positive number raise error
        if not isinstance(self.strike, float) and not isinstance(self.strike, int):
            raise TypeError('Strike price must be a number greater than 0')
        elif self.strike <= 0:
            raise ValueError('Strike price must be greater than 0')
        else:
            return self.strike

    # method to get risk-free rate
    def get_risk_free_rate(self):  # if risk-free rate is a number greater than 0 raise error
        if not isinstance(self.risk_free_rate, float) and not isinstance(self.risk_free_rate, int):
            raise TypeError('Risk-Free Rate must be a number')
        else:
            return self.risk_free_rate

    # method to get years to expiry
    def get_years_to_expiry(self):  # if days to expiry is not a positive number raise error
        if not isinstance(self.days_to_expiry, float) and not isinstance(self.days_to_expiry, int):
            raise TypeError('Days to Expiry must be a number greater than 0')
        elif self.days_to_expiry <= 0:
            raise ValueError('Days to Expiry must be positive')
        else:
            return self.days_to_expiry / 365

    # method to get option type
    def get_option_type(self):  # if option type is not put or call raise error
        if not isinstance(self.option_type, str):
            raise TypeError('Option Type must be -Put- or -Call-')
        elif self.option_type == 'Call' or self.option_type == 'Put':
            return self.option_type
        else:
            raise ValueError('Option Type must be -Put- or -Call-')

    # method to get model type
    def get_model_type(self):  # if model type is not BlackScholes or Bachelier
        if not isinstance(self.model_type, str):
            raise TypeError('Model type must be -Bachelier- or -BlackScholes-')
        elif self.model_type == 'Bachelier' or self.model_type == 'BlackScholes':
            return self.model_type
        else:
            raise ValueError('Model type must be -Bachelier- or -BlackScholes-')

    # method to calculate implied volatility given model and option types
    def calculate_volatility(self):

        # Newton-Raphson method is used - REPLACE THIS METHOD WITH THE SAME INPUTS FOR OTHER METHODS OF CALCULATION
        return conversion_newton_raphson(self.get_model_type(), self.get_option_type(), self.get_market_price(),
                                         self.get_underlying(), self.get_strike(), self.get_risk_free_rate(),
                                         self.get_years_to_expiry())

    # method get market price
    def get_market_price(self):
        if not isinstance(self.market_price, float) and not isinstance(self.market_price, int):
            raise TypeError('Market Price must be a number greater than 0')
        elif self.market_price <= 0:
            raise ValueError('Market Price must be greater than 0')
        else:
            return self.market_price


# ------------------------- FUNCTIONS USED FOR IMPLIED VOLATILITY CALCULATIONS -------------------------

# calculate Black-Scholes values to insert into normal cumulative distributions
def black_scholes_params(volatility, underlying_price, strike, risk_free, expiry):

    d1 = (1 / (volatility * sqrt(expiry))) * (log(underlying_price / strike) + (risk_free + ((volatility ** 2) / 2)) * expiry)
    d2 = d1 - volatility * sqrt(expiry)

    return d1, d2


# calculate vega value (derivative of Option price with respect to implied volatility)
def black_scholes_vega(volatility, underlying_price, strike, risk_free, expiry):

    d1, _ = black_scholes_params(volatility, underlying_price, strike, risk_free, expiry)
    vega = underlying_price * norm.pdf(d1) * sqrt(expiry)

    return vega


# determine price for Option using Black-Scholes
def black_scholes_price(option_type, volatility, underlying_price, strike, risk_free, expiry):

    d1, d2 = black_scholes_params(volatility, underlying_price, strike, risk_free, expiry)

    if option_type == 'Call':
        price = (norm.cdf(d1) * underlying_price) - (norm.cdf(d2) * strike * exp(-risk_free * expiry))
    elif option_type == 'Put':
        price = (norm.cdf(-d2) * strike * exp(-risk_free * expiry)) - (norm.cdf(-d1) * underlying_price)

    return price


# Brenner and Subrahmanyam volatility closed form approximation - can be used as first estimate for volatility
def bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry):

    if option_type == 'Call':
        return (market_price / underlying_price) * sqrt(2 * pi / expiry)
    elif option_type == 'Put':
        return ((underlying_price + market_price - strike * exp(-risk_free*expiry))/underlying_price) * sqrt(2 * pi / expiry)


# Newton-Raphson method for Black-Scholes volatility estimation - converts lognormal to normal volatility for Bachelier
def conversion_newton_raphson(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0
    volatility = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    # compute Newton-Raphson iterations until tolerance is reached
    while count < max_iterations:

        vega = black_scholes_vega(volatility, underlying_price, strike, risk_free, expiry)
        price = black_scholes_price(option_type, volatility, underlying_price, strike, risk_free, expiry)

        if vega == 0.0:
            print('Vega value of 0')
            return nan

        prev_vol = volatility  # store previous volatility estimate
        volatility = max(volatility - (price - market_price) / vega, 1e-10)  # calculate new volatility estimate according to NR method

        count += 1
        if round(prev_vol, decimal_places) == round(volatility, decimal_places):
            if model_type == 'Bachelier':  # convert to Bachelier volatility
                return lognormal_to_normal(volatility, underlying_price, strike, risk_free, expiry)
            elif model_type == 'BlackScholes':
                return volatility

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# function to convert lognormal (Black-Scholes) to normal (Bachelier) implied volatility {Hagan (1998), Viorel and Dan (2011)}
def lognormal_to_normal(volatility, underlying_price, strike, risk_free, expiry):

    forward = underlying_price * exp(risk_free * expiry)
    coef = volatility * ((forward - strike) / log(forward/strike))

    norm_vol = coef * (1 / (1 + (1/24)*(1-(1/120)*(log(forward/strike)**2))*(volatility**2)*expiry + (1/5760)*(volatility**4)*(expiry**2)))

    return norm_vol


# ------------------------- FUNCTIONS USED FOR IMPLIED VOLATILITY CALCULATIONS (not used in final application) -------------------------


# calculate Corrado and Miller volatility - can be used as first estimate for volatility
def cm_volatility(market_price, underlying_price, strike, risk_free, expiry):

    alpha = market_price - (underlying_price-strike*exp(-risk_free*expiry))/2
    beta = alpha + sqrt((alpha ** 2) - ((underlying_price - strike * exp(-risk_free*expiry))**2)/pi)
    volatility = (1 / sqrt(expiry)) * ((sqrt(2 * pi) / (underlying_price + strike * exp(risk_free * expiry))) + beta)

    return volatility


# Newton-Raphson method for Black-Scholes volatility estimation (non-convergence possible, higher speed)
def newton_raphson(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0
    if model_type == 'BlackScholes':
        volatility = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        # volatility = 0.5
    elif model_type == 'Bachelier':
        volatility = market_price * bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        # volatility = market_price * 0.5

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    # compute Newton-Raphson iterations until tolerance is reached
    while count < max_iterations:

        # determine vega (derivative of option price with respect to volatility) and price and return nan if vega is 0
        if model_type == 'BlackScholes':
            vega = black_scholes_vega(volatility, underlying_price, strike, risk_free, expiry)
            price = black_scholes_price(option_type, volatility, underlying_price, strike, risk_free, expiry)
        elif model_type == 'Bachelier':
            vega = bachelier_vega(volatility, underlying_price, strike, risk_free, expiry)
            price = bachelier_price(option_type, volatility, underlying_price, strike, risk_free, expiry)

        if vega == 0.0:
            print('Vega value of 0')
            return nan

        prev_vol = volatility  # store previous volatility estimate
        volatility = max(volatility - (price - market_price) / vega, 1e-10)  # calculate new volatility estimate according to NR method

        count += 1
        if round(prev_vol, decimal_places) == round(volatility, decimal_places):
            return volatility

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# Bachelier Option price
def bachelier_price(option_type, volatility, underlying_price, strike, risk_free, expiry):

    forward = underlying_price * exp(risk_free * expiry)  # determine forward price
    alpha = (forward - strike) / (volatility * sqrt(expiry))

    # determine theta value based on option type
    if option_type == 'Call':
        theta = 1
    elif option_type == 'Put':
        theta = -1

    # determine price based on option type and parameters
    price = volatility * sqrt(expiry) * norm.pdf(alpha) + theta * (forward - strike) * norm.cdf(theta * alpha)

    return price


# Bachelier vega value
def bachelier_vega(volatility, underlying_price, strike, risk_free, expiry):

    exponential = ((log(underlying_price)-log(strike) + 0.5*(volatility**2)*expiry)**2) / (2*(volatility**2)*expiry)
    vega = underlying_price * sqrt(expiry/(2*pi)) * exponential

    return vega


# Secant method for volatility estimation (non-convergence possible, higher speed)
def secant(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0
    if model_type == 'BlackScholes':
        initial_vol = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        vol1 = min((initial_vol + 0.05, 1))
        vol2 = max((initial_vol - 0.05, 0))
    elif model_type == 'Bachelier':
        initial_vol = market_price * bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        vol1 = max((initial_vol + 0.01, 0))
        vol2 = max((initial_vol - 0.01, 0))

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    while count < max_iterations:  # compute Secant iterations

        # determine prices for volatility estimates based on model type
        if model_type == 'BlackScholes':
            price1 = black_scholes_price(option_type, vol1, underlying_price, strike, risk_free, expiry)
            price2 = black_scholes_price(option_type, vol2, underlying_price, strike, risk_free, expiry)
        elif model_type == 'Bachelier':
            price1 = bachelier_price(option_type, vol1, underlying_price, strike, risk_free, expiry)
            price2 = bachelier_price(option_type, vol2, underlying_price, strike, risk_free, expiry)

        volatility = max(vol1 - price1 * ((vol1 - vol2) / (price1 - price2)), 1e-10)  # calculate new volatility estimate according to Secant method

        if round(vol1, decimal_places) == round(vol2, decimal_places) == round(volatility, decimal_places):  # check if estimate is at required tolerance
            return volatility

        count += 1
        vol2 = vol1
        vol1 = volatility  # update previous estimates for this iteration

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# Secant method for volatility estimation (non-convergence possible, higher speed) - converts lognormal to normal volatility for Bachelier
def conversion_secant(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0
    initial_vol = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
    vol1 = min((initial_vol + 0.05, 1))
    vol2 = max((initial_vol - 0.05, 0))

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    while count < max_iterations:  # compute Secant iterations

        # determine prices for volatility estimates based on model type
        price1 = black_scholes_price(option_type, vol1, underlying_price, strike, risk_free, expiry)
        price2 = black_scholes_price(option_type, vol2, underlying_price, strike, risk_free, expiry)

        volatility = max(vol1 - price1 * ((vol1 - vol2) / (price1 - price2)), 1e-10)  # calculate new volatility estimate according to Secant method

        if round(vol1, decimal_places) == round(vol2, decimal_places) == round(volatility, decimal_places):  # check if estimate is at required tolerance
            if model_type == 'BlackScholes':
                return volatility
            elif model_type == 'Bachelier':
                return lognormal_to_normal(volatility, underlying_price, strike, risk_free, expiry)

        count += 1
        vol2 = vol1
        vol1 = volatility  # update previous estimates for this iteration

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# False Position method for volatility estimation (guarantees convergence)
def false_position(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0
    if model_type == 'BlackScholes':
        initial_vol = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        vola = min((initial_vol + 0.01, 1))
        volb = max((initial_vol - 0.01, 0))
    elif model_type == 'Bachelier':
        initial_vol = market_price * bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
        vola = max((initial_vol + 0.01, 0))
        volb = max((initial_vol - 0.01, 0))

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    while count < max_iterations:  # compute False Positiion iterations

        # determine prices for volatility estimates based on model type
        if model_type == 'BlackScholes':
            price_a = black_scholes_price(option_type, vola, underlying_price, strike, risk_free, expiry)
            price_b = black_scholes_price(option_type, volb, underlying_price, strike, risk_free, expiry)
        elif model_type == 'Bachelier':
            price_a = bachelier_price(option_type, vola, underlying_price, strike, risk_free, expiry)
            price_b = bachelier_price(option_type, volb, underlying_price, strike, risk_free, expiry)

        volatility = max(volb - price_b * ((volb - vola) / (price_b - price_a)), 1e-10)  # calculate new volatility estimate according to FP method

        if round(vola, decimal_places) == round(volb, decimal_places) == round(volatility, decimal_places):  # check if estimate is at required tolerance
            return volatility

        # update values for next iteration based on signs of prices
        price = black_scholes_price(option_type, volatility, underlying_price, strike, risk_free, expiry)
        if (price >= 0 and price_a >= 0) or (price < 0 and price_a < 0):
            vola = volatility
        else:
            volb = volatility

    count += 1

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# False Position method for volatility estimation (guarantees convergence - converts lognormal to normal volatility for Bachelier
def conversion_false_position(model_type, option_type, market_price, underlying_price, strike, risk_free, expiry, tolerance=1e-8, max_iterations=100000):

    # initialise iteration count and initial volatility estimation
    count = 0

    initial_vol = bren_sub_volatility(option_type, market_price, underlying_price, strike, risk_free, expiry)
    vola = min((initial_vol + 0.01, 1))
    volb = max((initial_vol - 0.01, 0))

    # determine number of decimal places tolerance requires
    decimal_places = abs(decimal.Decimal(str(tolerance)).as_tuple().exponent)

    while count < max_iterations:  # compute False Positiion iterations

        price_a = black_scholes_price(option_type, vola, underlying_price, strike, risk_free, expiry)
        price_b = black_scholes_price(option_type, volb, underlying_price, strike, risk_free, expiry)

        volatility = max(volb - price_b * ((volb - vola) / (price_b - price_a)), 1e-10)  # calculate new volatility estimate according to FP method

        if round(vola, decimal_places) == round(volb, decimal_places) == round(volatility, decimal_places):  # check if estimate is at required tolerance
            if model_type == 'BlackScholes':
                return volatility
            elif model_type == 'Bachelier':
                return lognormal_to_normal(volatility, underlying_price, strike, risk_free, expiry)

        # update values for next iteration based on signs of prices
        price = black_scholes_price(option_type, volatility, underlying_price, strike, risk_free, expiry)
        if (price >= 0 and price_a >= 0) or (price < 0 and price_a < 0):
            vola = volatility
        else:
            volb = volatility

    count += 1

    # if max iterations is reached return nan
    print('Maximum iterations (' + str(max_iterations) + ') reached')
    return nan


# MAIN FUNCTION
def main():

    market_data = pd.read_csv('input.csv')  # read in input csv file

    # remove all occurrences of a duplicated trade ID to ensure all trades can be uniquely identified
    market_data.drop_duplicates(subset='ID', keep=False, inplace=True)

    volatility_data = []  # initialise storage list for output data

    timer = time.time()  # initialise timer

    for i, trade_data in market_data.iterrows():  # iterate through trade data

        trade = Trade(trade_data)  # create trade object for a particular trade

        print('Completion: ' + str(round((i/len(market_data.index))*100, 4)) + '% ~ Trade: ' + str(trade.get_id()) + ' of '
              + str(len(market_data.index)) + ' ~ Runtime: ' + str(dt.timedelta(seconds=round((time.time() - timer), 0))))

        try:  # add volatility data to storage list if input is valid
            volatility_data.append([trade.get_id(), trade.get_spot(), trade.get_strike(), trade.get_risk_free_rate(),
                                    trade.get_years_to_expiry(), trade.get_option_type(), trade.get_model_type(),
                                    trade.calculate_volatility(), trade.get_market_price()])
        except (TypeError, ValueError) as error:
            print(error)  # skip trade if input is invalid and display error
            continue

    # create a dtaaframe with volatility data
    output_data = pd.DataFrame(data=volatility_data, columns=['ID', 'Spot', 'Strike', 'Risk-Free Rate',
                                                              'Years To Expiry', 'Option Type', 'Model Type',
                                                              'Implied Volatility', 'Market Price'])
    output_data.to_csv('output.csv')  # output volatility dataframe to csv file


# run main function
if __name__ == '__main__':
    main()
