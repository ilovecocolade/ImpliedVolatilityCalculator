import unittest
from imc import Trade
import pandas as pd


# unit testing class to validate all functionality of Trade class apart from volatility calculations
class TestIMC(unittest.TestCase):

    # test get_id method
    def test_get_id(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756,  'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_id(), 1)

        trade1 = Trade(pd.Series(data=['id', 'Stock', 2.987, -0.02, 47.78, 0.756,  'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(TypeError):
            trade1.get_id()

        trade2 = Trade(pd.Series(data=[-1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade3 = Trade(pd.Series(data=[1.065, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade2.get_id()
            trade3.get_id()

    # test get_underlying method
    def test_get_underlying(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2, -0.02, '47.78', '0.756', 'Call', 'Bachelier', '0.547'],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_underlying(), 2.987)
        self.assertEqual(trade1.get_underlying(), 2)

        trade2 = Trade(pd.Series(data=[1, 'Stock', 'price', -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(TypeError):
            trade2.get_underlying()

        trade3 = Trade(pd.Series(data=[1, 'Stock', -1, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(ValueError):
            trade3.get_underlying()

    # test get_spot method
    def test_get_spot(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Future', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_spot(), 0.547)
        self.assertEqual(trade1.get_spot(), 2.987)

        trade2 = Trade(pd.Series(data=[1, 'ETF', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(ValueError):
            trade2.get_spot()

        trade3 = Trade(pd.Series(data=[1, 0, 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(TypeError):
            trade3.get_spot()

    # test get_strike method
    def test_get_strike(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 30, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_strike(), 0.756)
        self.assertEqual(trade1.get_strike(), 30)

        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 'strike', 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(TypeError):
            trade2.get_strike()

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade4 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, -0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade3.get_strike()
            trade4.get_strike()

    # test get_risk_free_rate method
    def test_risk_free_rate(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, 0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, 0, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_risk_free_rate(), -0.02)
        self.assertEqual(trade1.get_risk_free_rate(), 0.02)
        self.assertEqual(trade2.get_risk_free_rate(), 0)

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, 'risk', 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        with self.assertRaises(TypeError):
            trade3.get_risk_free_rate()

    # test get_years_to_expiry method
    def test_get_years_to_expiry(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_years_to_expiry(), 47.78/365)
        self.assertEqual(trade1.get_years_to_expiry(), 47/365)

        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 'expiry', 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(TypeError):
            trade2.get_years_to_expiry()

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, -1, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade3.get_years_to_expiry()

    # test get_option_type method
    def test_get_option_type(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Put', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_option_type(), 'Call')
        self.assertEqual(trade1.get_option_type(), 'Put')

        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 1, 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(TypeError):
            trade2.get_option_type()

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Bond', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade3.get_option_type()

    # test get_model_type method
    def test_get_model_type(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'BlackScholes', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_model_type(), 'Bachelier')
        self.assertEqual(trade1.get_model_type(), 'BlackScholes')

        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'SABR', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade2.get_model_type()

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 1, 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(TypeError):
            trade3.get_model_type()

    # test get_market_price method
    def test_get_market_price(self):
        trade0 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))
        trade1 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 1],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        self.assertEqual(trade0.get_market_price(), 0.547)
        self.assertEqual(trade1.get_market_price(), 1)

        trade2 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', -0.547],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(ValueError):
            trade2.get_market_price()

        trade3 = Trade(pd.Series(data=[1, 'Stock', 2.987, -0.02, 47.78, 0.756, 'Call', 'Bachelier', 'price'],
                                 index=['ID', 'Underlying Type', 'Underlying', 'Risk-Free Rate', 'Days To Expiry',
                                        'Strike', 'Option Type', 'Model Type', 'Market Price']))

        with self.assertRaises(TypeError):
            trade3.get_market_price()


if __name__ == '__main__':
    unittest.main()
