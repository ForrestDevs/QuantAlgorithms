# Author: Luke Gannon, 
# Date: October 2022 
# QUANTT Team

# region imports
import numpy as np
import pandas as pd
import QuantConnect.Securities.Future
from AlgorithmImports import *
from datetime import timedelta
# endregion

# region CONSTATNS
OPT_1_STP = 0.55
OPT_1_ERT = 0.50
OPT_1_SSL = 0.05
# endregion CONSTANTS



def RoundToTick(val):
    return round(val * 4) / 4


class ADRIv2(QCAlgorithm):
    """
    The ADRIv2 (AlgorithmicDefiningRangeInterval). This algorithm has an intraday profile with 3 active sessions; known first hand as ODR RDR ADR.
    Only one entry per session is allowed, filtering for minimum of 1:1RR. Parameters to be added....
    """

    def Initialize(self):
        # region Environment Settings
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 3, 2)
        self.SetCash(10000000)
        self.SetWarmUp(timedelta(days=1))
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, accountType=AccountType.Margin)
        # endregion Environment Settings

        # region Logging Control
        self.enable_logging = False
        self.enable_fill_logging = True
        self.enable_EOD_logging = False
        self.enable_phase_1_logging = True
        self.announce_p1 = False
        self.enable_phase_2_logging = True
        self.announce_p2 = False
        self.enable_phase_3_logging = True
        self.announce_p3 = False
        self.enable_phase_4_logging = True
        self.announce_p4 = False
        self.announce_daily_high_impact = False
        self.announce_himpactnews = False
        self.enable_contract_ordersize_logging = True
        self.enable_risk_verification_logging = True
        self.enable_session_confirmation_logging = True
        # endregion Logging Control

        # region Tradable Assets
        self.contract_openinterest = 100
        # Initialize ES futures contract
        self.es = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute, extendedMarketHours=True)
        # Set contract expiry - return contracts that expire within 'self.contractExpiryDays' days
        self.es.SetFilter(0, 90)
        # Need to have a 5-minute bar that fires every 5 minute to run func Check5mBarCloses
        self.es_5m_bars = self.Consolidate(self.es.Symbol, timedelta(minutes=5), self.Check5mBarCloses)
        # endregion Tradeable Assets

        # region Events/Data
        # Pandas Dataframe of entire Economic Calendar
        self.economic_calendar = self.DownloadEconomicCalendar()
        # Scheduled Events to Update ODR RDR ADR
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=4, minute=0), self.UpdateODR)
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=8, minute=30), self.EndODR)
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=10, minute=30), self.UpdateRDR)
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=16, minute=0), self.EndRDR)
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=20, minute=30), self.UpdateADR)
        self.Schedule.On(self.DateRules.EveryDay(self.es.Symbol), self.TimeRules.At(hour=2, minute=0), self.EndADR)
        # endregion Events/Data

        # region Indicators
        # Rolling MAX and MIN for use with the DR
        self.dr_max = self.MAX(self.es.Symbol, 60, Resolution.Minute, selector=Field.High)
        self.dr_min = self.MIN(self.es.Symbol, 60, Resolution.Minute, selector=Field.Low)
        # Rolling MAX and MIN for use with IDR on Consolidated 5 minute bar
        # Careful with another bug found here, was set at 60 period, however this receives 5m bar resolution.
        # 60 periods x 5m bars would be 5 hours a different look back than the 1-hour intended range with the DR.
        # Better set to 12 bars instead 60/5 = 12 periods would be an hour, the same as the DR now.
        self.idr_max_opens, self.idr_max_closes, self.idr_min_opens, self.idr_min_closes = Maximum(12), Maximum(
            12), Minimum(12), Minimum(12)
        self.RegisterIndicator(self.es.Symbol, self.idr_max_opens, self.es_5m_bars, selector=Field.Open)
        self.RegisterIndicator(self.es.Symbol, self.idr_max_closes, self.es_5m_bars, selector=Field.Close)
        self.RegisterIndicator(self.es.Symbol, self.idr_min_opens, self.es_5m_bars, selector=Field.Open)
        self.RegisterIndicator(self.es.Symbol, self.idr_min_closes, self.es_5m_bars, selector=Field.Close)
        # endregion Indicators

        # region Session Variables
        self.current_session = None
        self.entry_models_initialized = False
        self.can_trade = False
        
        # This becomes true once the DR is finished being printed
        self.session_dr_phase_1 = False
        # This becomes true once the first directional signal is present
        self.session_dr_phase_2 = False
        # This becomes true when a second directional signal is present
        self.session_dr_phase_3 = False
        # This becomes true when 0.5 std is hit on either side of the DR
        self.session_dr_phase_4 = False

        self.session_dr_direction = 0
        self.session_dr_range, self.session_dr_high, self.session_dr_low, self.session_dr_mid = 0, 0, 0, 0
        self.session_dr_half_std_high = 0
        self.session_dr_half_std_low = 0
        self.session_idr_direction = 0
        self.session_idr_range, self.session_idr_high, self.session_idr_low, self.session_idr_mid = 0, 0, 0, 0
        self.session_idr_half_std_high = 0
        self.session_idr_half_std_low = 0
        # endregion Session Variables

        # region DR/IDR Variables
        # Overnight Defining Range Holding Variables
        self.odr_open, self.odr_range, self.odr_mid, self.odr_high, self.odr_low, self.odr_direction = 0, 0, 0, 0, 0, 0
        # Overnight Implied Defining Range Holding Variables
        self.oidr_range, self.oidr_mid, self.oidr_high, self.oidr_low, self.oidr_direction = 0, 0, 0, 0, 0
        # Real Defining Range Holding Variables
        self.rdr_open, self.rdr_range, self.rdr_mid, self.rdr_high, self.rdr_low, self.rdr_direction = 0, 0, 0, 0, 0, 0
        # Real Implied Defining Range Holding Variables
        self.ridr_range, self.ridr_mid, self.ridr_high, self.ridr_low, self.ridr_direction = 0, 0, 0, 0, 0
        # After Defining Range Holding Variables
        self.adr_open, self.adr_range, self.adr_mid, self.adr_high, self.adr_low, self.adr_direction = 0, 0, 0, 0, 0, 0
        # After Implied Defining Range Holding Variables
        self.aidr_range, self.aidr_mid, self.aidr_high, self.aidr_low, self.aidr_direction = 0, 0, 0, 0, 0
        # endregion DR/IDR Variables

        # region Trade Parameters
        # Maximum risk allowed for a single entry model.
        self.position_max_risk = 0.05
        # Minimum Risk to Reward required to verify trade. All setups less than this value will be discarded.
        self.minimum_rr = 0.7
        # IDR STD, sets the takeprofit target for the session.
        self.std_tp = 0.55
        # IDR STD, sets the stoploss target for the session.
        self.std_sl = 0.05
        # Retracement % back into the IDR range for entry in direction of first signal
        self.retracement_percent = 0.75
        # Control when open orders should be closed. True for start of session, false for end of session.
        self.cancel_orders_on_open = True
        # Enable or disable specific sessions, or trade directions.
        self.enable_odr_entries = True
        self.enable_rdr_entries = True
        self.enable_adr_entries = True
        self.enable_short_entries = True
        self.enable_long_entries = True
        # Single fire, end of session vars
        self.rdr_over = False
        self.odr_over = False
        self.adr_over = False
        self.rdr_start = False
        self.odr_start = False
        self.adr_start = False
        # endregion Trade Parameters

        # region Session Entry, SL, TP Price Variables
        self.session_takeprofit = 0
        self.session_stoploss = 0
        self.session_short_entry_price = 0
        self.session_long_entry_price = 0
        self.session_entry_ticket = None
        self.session_stoploss_ticket = None
        self.session_takeprofit_ticket = None
        # endregion Session Entry, SL, TP Price Variables

        # region Statistic Variables
        self.total_entries = 0
        self.total_exits_sl = 0
        self.total_exits_tp = 0
        self.total_exits_eos = 0
        self.session_long_fails = 0
        self.session_short_fails = 0
        # endregion Statistic Variables

    def Check5mBarCloses(self, bar) -> None:
        b_open, b_high, b_low, b_close = bar.Open, bar.High, bar.Low, bar.Close

        if self.enable_logging:
            self.Log(
                f"{self.Time}: 5m bar update."
                f"sDRp1: {self.session_dr_phase_1}, "
                f"sDRp2: {self.session_dr_phase_2}, "
                f"sDRp3: {self.session_dr_phase_3}, "
                f"sDRp4: {self.session_dr_phase_4}, "
                f"sDRdirec: {self.session_dr_direction}"
            )

        # Only do this check while the DR has been completed for the session.
        if self.session_dr_phase_1:
            # If no signal has been generated for the session lets be alert for the first one.
            if not self.session_dr_phase_2:
                if b_close > self.session_dr_high:
                    if self.enable_session_confirmation_logging:
                        self.Log(f"{self.current_session} Session direction confirmed! Look for Longs!")
                    self.session_dr_phase_2 = True
                    self.session_dr_direction = 1
                elif b_close < self.session_dr_low:
                    if self.enable_session_confirmation_logging:
                        self.Log(f"{self.current_session}Session direction confirmed! Look for Shorts!")
                    self.session_dr_phase_2 = True
                    self.session_dr_direction = -1
            if not self.session_dr_phase_3 and self.session_dr_phase_2:  # Phase 2 currently True lets stay alert for a DR fail or Phase 4 alert.
                # Session direction gave long signal, but we closed below DR low, count failure and enable phase 3.
                if self.session_dr_direction == 1 and b_close < self.session_dr_low:
                    self.session_dr_phase_3 = True
                    self.session_long_fails += 1
                # Session direction gave short signal, but we closed above DR high, count failure and enable phase 3.
                if self.session_dr_direction == -1 and b_close > self.session_dr_high:
                    self.session_dr_phase_3 = True
                    self.session_short_fails += 1
                # Session low-hanging fruit acquired
                if b_close >= self.session_dr_half_std_high:
                    self.session_dr_phase_4 = True
                if b_close <= self.session_dr_half_std_low:
                    self.session_dr_phase_4 = True

    # region News Functions
    def DownloadEconomicCalendar(self):
        """
        Downloads an internet hosted CSV file containing the entire economic calendar, sourced using a proprietary API.
        Returns a pandas dataframe initialized from the CSV file.
        """
        url = 'https://github.com/Lukester45/EconomicCalendarAPI/blob/main/news_event_data1.csv?raw=true'
        try:
            data = self.Download(url).split('\n')
            date = [x.split(',')[0] for x in data][1:]
            time = [x.split(',')[1] for x in data][1:]
            currency = [x.split(',')[2] for x in data][1:]
            impact = [x.split(',')[3] for x in data][1:]
            event = [x.split(',')[4] for x in data][1:]
            d = {'date': date, 'time': time, 'currency': currency, 'event': event, 'impact': impact}
            df = pd.DataFrame(d)

            event_titles = ["CPI m/m", "CPI q/q", "CPI y/y", "Core CPI m/m", "Core CPI y/y",
                            "Non-Farm Employment Change", "FOMC Meeting Minutes"]

            all_usd_high_impact = df[(df['currency'] == "USD") & (df['impact'] == "High")]
            usd_HI_NFP_CPI_FOMC = df[
                (df['currency'] == "USD") & (df['impact'] == "High") & (df['event'].isin(event_titles))]

            return usd_HI_NFP_CPI_FOMC
        except ValueError:
            self.Debug("Error Downloading Economic Calendar")

    def CheckEconomicImpact(self, date):
        return True
        # time_range_odr = pd.date_range(start=datetime(2020, 1, 1, 4, 0, 0), end=datetime(2020, 1, 1, 8, 30, 0),
        #                                freq='min').time
        # time_range_rdr = pd.date_range(start=datetime(2020, 1, 1, 9, 30, 0), end=datetime(2020, 1, 1, 16, 0, 0),
        #                                freq='min').time
        # time_range_adr = pd.date_range(start=datetime(2020, 1, 1, 20, 30, 0), end=datetime(2020, 1, 1, 2, 0, 0),
        #                                freq='min').time
        # formatted_date = datetime.strftime(date, '%b %-d %Y')
        # formatted_time = datetime.strftime(date, '%I:%M %p')
        # rtn = True
        #
        # self.Log(
        #     "Checking Economic Impact"
        #     f"Current time: {formatted_time}"
        #     f"Current date: {formatted_date}"
        # )
        #
        # if not self.announce_daily_high_impact and formatted_date in self.economic_calendar['date'].values:
        #     self.announce_daily_high_impact = True
        #     self.Log("High impact news for for todays trading day!")
        #
        #     todays_news = self.economic_calendar[self.economic_calendar['date'] == formatted_date]
        #     for time in todays_news['time']:
        #         if time == 'All Day':
        #             self.Log("All Day High Impact News!")
        #             rtn = False
        #         else:
        #             if self.current_session == 'ODR' and time in time_range_odr:
        #                 self.Log("High Impact News for ODR Session, no trades will be executed.")
        #                 rtn = False
        #             if self.current_session == 'RDR' and time in time_range_rdr:
        #                 self.Log("High Impact News for RDR Session, no trades will be executed.")
        #                 rtn = False
        #             if self.current_session == 'ADR' and time in time_range_adr:
        #                 self.Log("High Impact News for ADR Session, no trades will be executed.")
        #                 rtn = False
        #
        # return rtn

    # endregion News Functions

    # region Reset Variable Functions
    def EndOfDailyRange(self):
        """
        Called at the end of the ADR trading session, computes statistics for the day.
        """
        self.ResetDailyTradeParams()
        self.announce_daily_high_impact = False

    def ResetDailyTradeParams(self):
        if self.enable_EOD_logging:
            self.Log(f"End of daily trading session, open_position? {self.Portfolio[self.es.Symbol].Quantity}")
        # Resetting all DR/IDR Session Variables back to 0
        # region DR/IDR Variables
        # Overnight Defining Range Holding Variables
        self.odr_open, self.odr_range, self.odr_mid, self.odr_high, self.odr_low, self.odr_direction = 0, 0, 0, 0, 0, 0
        # Overnight Implied Defining Range Holding Variables
        self.oidr_range, self.oidr_mid, self.oidr_high, self.oidr_low, self.oidr_direction = 0, 0, 0, 0, 0
        # Real Defining Range Holding Variables
        self.rdr_open, self.rdr_range, self.rdr_mid, self.rdr_high, self.rdr_low, self.rdr_direction = 0, 0, 0, 0, 0, 0
        # Real Implied Defining Range Holding Variables
        self.ridr_range, self.ridr_mid, self.ridr_high, self.ridr_low, self.ridr_direction = 0, 0, 0, 0, 0
        # After Defining Range Holding Variables
        self.adr_open, self.adr_range, self.adr_mid, self.adr_high, self.adr_low, self.adr_direction = 0, 0, 0, 0, 0, 0
        # After Implied Defining Range Holding Variables
        self.aidr_range, self.aidr_mid, self.aidr_high, self.aidr_low, self.aidr_direction = 0, 0, 0, 0, 0
        # endregion DR/IDR Variables

    def ResetSessionTradeParams(self):
        """
        This class method is only once called at the end of any trading session.
        """
        self.current_session = 'QC'
        # Reset the session DR levels & phases
        self.session_dr_direction, self.session_dr_range, self.session_dr_high, self.session_dr_low, self.session_dr_mid = 0, 0, 0, 0, 0
        self.session_idr_direction, self.session_idr_range, self.session_idr_high, self.session_idr_low, self.session_idr_mid = 0, 0, 0, 0, 0
        self.session_dr_phase_1, self.session_dr_phase_2, self.session_dr_phase_3, self.session_dr_phase_4 = False, False, False, False
        # Reset entry model initialized
        self.entry_models_initialized = False
        # Reset the session entry prices
        self.session_long_entry_price = 0
        self.session_short_entry_price = 0
        # Reset can trade parameter
        self.can_trade = True

        self.announce_p1 = False
        self.announce_p2 = False
        self.announce_p3 = False
        self.announce_p4 = False
        self.announce_himpactnews = False
        # Reset the session trade ticket
        self.session_entry_ticket = None
        self.session_stoploss_ticket = None
        self.session_takeprofit_ticket = None

    def ResetTradeParams(self):
        self.session_stoploss = 0
        self.session_takeprofit = 0
        self.session_long_entry_price = 0
        self.session_long_entry_price = 0
        self.session_entry_ticket = None

    # endregion Reset Variable Functions

    # region Order Management Functions
    def ContractOrderSize(self, symbol, direction, entry, stoploss_price):
        # region Initialize Base Variables
        sl_range = ((entry - stoploss_price) * direction)
        # Amount of cash we're willing to risk
        cash_risk = self.Portfolio.TotalPortfolioValue * self.position_max_risk
        # Maximum number of orders that we can open for this position
        max_orders = self.CalculateOrderQuantity(symbol, direction)
        # The dollar value of 1 tick per 1 contract
        tick_contract_value = self.es.SymbolProperties.ContractMultiplier
        # The total dollar risk for the position using max orders
        max_dollar_risk = (max_orders * direction) * (tick_contract_value * sl_range)
        # Number of contracts to order
        order_size = max_orders
        # endregion Initialize Base Variables

        if self.enable_contract_ordersize_logging:
            self.Log(
                f"Contract Order Size Function Initial Values: "
                f"slR: {sl_range}, "
                f"direction: {direction}, "
                f"maxO: {max_orders}, "
                f"tickCV: {tick_contract_value}, "
                f"maxDR: {max_dollar_risk}, "
                f"cashR: {cash_risk}, "
                f"oS: {order_size} "
            )

        while max_dollar_risk > cash_risk:
            self.Log(
                f"Max risk exceeds cash risk, lowering ordersize by 1. cO: {order_size}, mR: {max_dollar_risk}, cR: {cash_risk}")
            # Simple loop that continues to lower the order size by 1 until the cash risk is satisfied.

            if direction == 1 and order_size > 0:
                order_size -= 1
            elif direction == 1 and order_size == 0:
                order_size = 0
                max_dollar_risk = 0
                self.Log("Long position canceled, sl to large or not enough margin to open position.")

            if direction == -1 and order_size < 0:
                order_size += 1
            elif direction == -1 and order_size == 0:
                order_size = 0
                max_dollar_risk = 0
                self.Log("Short position canceled, sl to large or not enough margin to open position.")

            max_dollar_risk = (order_size * direction) * (tick_contract_value * sl_range)
            self.Log(f"cO: {order_size}, mR: {max_dollar_risk}, cR: {cash_risk}")

        # region Error Handling
        # If the order size is less than 0 we would enter a sell when we are in a long model
        # This is a huge error and better to just return 0 and log the error than to try and take the trade.
        if direction == 1:
            if order_size < 0:
                order_size = 0
                if self.enable_contract_ordersize_logging:
                    self.Log("Error with contract order size. Attempted to take a SHORT when LONG was intended.")
        if direction == -1:
            if order_size > 0:
                order_size = 0
                if self.enable_contract_ordersize_logging:
                    self.Log("Error with contract order size. Attempted to take a LONG when SHORT was intended.")
        # endregion Error Handling

        if self.enable_contract_ordersize_logging:
            self.Log(
                "Successfully Returning Contract Order Size "
                f"Account Value: {self.Portfolio.TotalPortfolioValue}, "
                f"Cash Risk: {cash_risk}$, "
                f"Order size: {order_size}, "
                f"for position entry: {direction}, "
                f"sLr: {sl_range}"
            )

        return order_size

    def VerifyRR(self, entry_price, sl_price, tp_price, direction):
        """
        Class method used to verify the session trade parameters and make sure that everything lines up.
        Usually called before initializing and entry model and also a chance to filter for certain RR setups.

        :param entry_price: float
        :param sl_price: float
        :param tp_price: float
        :param direction: float
        :return: Bool - True when the range is large enough and RR filter satisfied.
        """
        sl_range = 0
        tp_range = 0

        rtn = True

        if self.enable_risk_verification_logging:
            self.Log(f"entry: {entry_price}, SLp: {sl_price}, TPp: {tp_price}, direction: {direction}")

        if direction == 1:
            sl_range = (entry_price - sl_price)
            tp_range = (tp_price - entry_price)
        elif direction == -1:
            sl_range = (sl_price - entry_price)
            tp_range = (entry_price - tp_price)

        b_or_s = ""
        if direction == 1:
            b_or_s = "Buy"
        if direction == -1:
            b_or_s = "Sell"

        if sl_range == 0:
            rtn = False

        if tp_range == 0:
            rtn = False

        trade_ratio = 0

        if tp_range != 0 and sl_range != 0:
            trade_ratio = tp_range / sl_range

        if trade_ratio != 0 and trade_ratio < self.minimum_rr:
            rtn = False

        self.Log(
            f"Checking RR: {trade_ratio}, "
            f"minRR: {self.minimum_rr} "
            f"direction: {b_or_s}, "
            f"slr: {sl_range}, "
            f"tpr: {tp_range}"
        )

        if self.enable_risk_verification_logging:
            self.Log(f"Returning {rtn} from trade verification")

        return rtn

    def OnMarginCallWarning(self):
        self.Error("You received a margin call warning!")

    def ClosePosition(self, symbol, reason):
        if not self.IsWarmingUp:
            if self.Portfolio[symbol].Quantity != 0:
                self.total_exits_eos += 1
            self.session_stoploss = 0
            self.session_takeprofit = 0
            self.MarketOrder(symbol, (self.Portfolio[symbol].Quantity * -1), False, reason)
        else:
            if self.enable_logging:
                self.Log("Error: Algo warming up, no orders should be submitted!")

    def CancelPendingOrders(self, symbol):
        """
        Class method used to look for any pending, unfilled orders and cancel all of them.
        Usually called at the end of a trading session, to ensure no leftover orders.
        """
        if not self.IsWarmingUp:
            if len(self.Transactions.GetOpenOrders(symbol)) > 0:
                for x in self.Transactions.GetOpenOrders(symbol):
                    if self.enable_logging:
                        self.Log(f"{self.Time}: Pending Order Found! Canceling at End of Session")
                    order_tag = f"{self.Time}: Canceling pending unfilled order: {x.Id}. Reason: EoS"
                    self.Transactions.CancelOrder(x.Id, order_tag)
        else:
            if self.enable_logging:
                self.Log("Error: Algo warming up, no orders should be submitted!")

    def CheckOpenPosition(self, symbol, close):
        """
        Class method to check if there are currently any open positions and manages them accordingly.
        Usually called at the end of a trading session.

        :param symbol: QuantConnect.Securities.Future.Future.Symbol - Symbol for the asset.
        :param close: TradeBar.Close - float.
        :return: Void
        """
        self.ClosePosition(symbol, "EOS")
        # if self.enable_close_before_sessions:
        #     if self.Portfolio[symbol].Quantity != 0:
        #         if not self.session_dr_phase_3 and self.session_dr_direction == 1 and close > self.session_long_entry_price:
        #             if (close - self.session_long_entry_price) / (self.session_long_entry_price - self.current_sl) >= 1:
        #                 self.ClosePosition(symbol, f"Closing long In Profit @ price: {close}, Reason: EOS +1R")
        #             else:
        #                 self.current_sl = self.session_long_entry_price
        #         elif not self.session_dr_phase_3 and self.session_dr_direction == -1 and close < self.session_short_entry_price:
        #             if (self.session_short_entry_price - close) / (
        #                     self.current_sl - self.session_short_entry_price) >= 1:
        #                 self.ClosePosition(symbol, f"Close short in Profit @ price: {close}, Reason EOS +1R")
        #             else:
        #                 self.current_sl = self.session_short_entry_price
        #         elif not self.session_dr_phase_3 and self.session_dr_direction == 1 and close < self.session_long_entry_price:
        #             self.ClosePosition(symbol, f"Close long for loss @ price: {close}, Reason: EOS")
        #         elif not self.session_dr_phase_3 and self.session_dr_direction == -1 and close > self.session_short_entry_price:
        #             self.ClosePosition(symbol, f"Close short for loss @ price: {close}, Reason: EOS")

    # endregion Order Management Functions

    # region ScheduledEventHandlers
    def UpdateODR(self):
        self.current_session = 'ODR'
        self.GetSessionValues()
        self.odr_start = True

    def UpdateRDR(self):
        self.current_session = 'RDR'
        self.GetSessionValues()
        self.rdr_start = True

    def UpdateADR(self):
        self.current_session = 'ADR'
        self.GetSessionValues()
        self.adr_start = True

    def EndODR(self):
        # Activate single fire variable  to check open and pending orders and handle them accordingly
        self.odr_over = True

    def EndRDR(self):
        # Activate single fire variable  to check open and pending orders and handle them accordingly
        self.rdr_over = True

    def EndADR(self):
        # Activate single fire variable to check open and pending orders and handle them accordingly
        self.adr_over = True

    # endregion ScheduledEventHandlers

    def GetSessionValues(self) -> None:
        """
        This class method is called at end of the DR range for the start of the trading session.
        It calculates and sets the needed DR/IDR session values and checks for phase_1 to be complete.
        """

        self.session_dr_high = RoundToTick(self.dr_max.Current.Value)
        self.session_dr_low = RoundToTick(self.dr_min.Current.Value)
        self.session_idr_high = RoundToTick(max(self.idr_max_opens.Current.Value, self.idr_max_closes.Current.Value))
        self.session_idr_low = RoundToTick(min(self.idr_min_opens.Current.Value, self.idr_min_closes.Current.Value))
        self.session_dr_range = RoundToTick(self.session_dr_high - self.session_dr_low)
        self.session_idr_range = RoundToTick(self.session_idr_high - self.session_idr_low)
        self.session_dr_half_std_high = RoundToTick(self.session_dr_high + (self.session_dr_range / 2))
        self.session_dr_half_std_low = RoundToTick(self.session_dr_low - (self.session_dr_range / 2))
        self.session_idr_half_std_high = RoundToTick(self.session_dr_high + (self.session_idr_range / 2))
        self.session_idr_half_std_low = RoundToTick(self.session_dr_low - (self.session_idr_range / 2))

        if self.current_session == 'ODR':
            self.odr_high = self.session_dr_high
            self.odr_low = self.session_dr_low
            self.odr_range = self.session_dr_range
            self.oidr_high = self.session_idr_high
            self.oidr_low = self.session_idr_low
            self.oidr_range = self.session_idr_range
        elif self.current_session == 'RDR':
            self.rdr_high = self.session_dr_high
            self.rdr_low = self.session_dr_low
            self.rdr_range = self.session_dr_range
            self.ridr_high = self.session_idr_high
            self.ridr_low = self.session_idr_low
            self.ridr_range = self.session_idr_range
        elif self.current_session == 'ADR':
            self.adr_high = self.session_dr_high
            self.adr_low = self.session_dr_low
            self.adr_range = self.session_dr_range
            self.aidr_high = self.session_idr_high
            self.aidr_low = self.session_idr_low
            self.aidr_range = self.session_idr_range

        if self.session_dr_range > 0:
            self.session_dr_phase_1 = True
            if self.enable_logging:
                self.Log(
                    f"{self.current_session} Session DR Initialized! "
                    f"Session Phase 1 Enabled, with session values; "
                    f"DR_high: {self.session_dr_high}, "
                    f"DR_low: {self.session_dr_low}."
                )
        else:
            self.session_dr_phase_1 = False
            if self.enable_logging:
                self.Log(f"Error {self.current_session} session DR less than 0, Session Phase 1 Disabled!")

    def OnData(self, slice: QuantConnect.Data.Slice) -> None:
        """
        Event - v3.0 DATA EVENT HANDLER: (Pattern) Basic template for user to override for receiving all subscription data in a single event
        :param slice: The current slice of data keyed by symbol string
        """

        # Current Temporal Variables
        self.symbol, self.bar_close, self.bar_open, self.bar_high, self.bar_low = None, None, None, None, None

        # Let's find the most liquid contract from the continuous future chain for our current temporal position
        for chain in slice.FutureChains:
            # Look for contracts that only have a certain open interest
            self.popularContacts = [contract for contract in chain.Value if
                                    contract.OpenInterest > self.contract_openinterest]
            # If there are none return
            if len(self.popularContacts) == 0:
                return
            # Sort the contracts, high to low, by open interest
            sortedByOIContracts = sorted(self.popularContacts, key=lambda k: k.OpenInterest, reverse=True)
            self.liquidContract = sortedByOIContracts[0]  # The current most liquid contract
            self.symbol = self.liquidContract.Symbol  # Current trading contract's symbol
            # Initialize candle data
            tradebars = chain.Value.TradeBars
            # Add data to candle variables
            if self.symbol in tradebars.Keys:
                tradebar = tradebars[self.symbol]
                self.bar_close = tradebar.Close
                self.bar_open = tradebar.Open
                self.bar_high = tradebar.High
                self.bar_low = tradebar.Low

        # If the tradebar data is not loaded yet, return the function
        if self.bar_close is None and self.bar_open is None and self.bar_high is None and self.bar_low is None:
            return

        # region Beginning Of New Session Management
        if self.odr_start:
            self.odr_start = False
            if self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
        if self.rdr_start:
            self.rdr_start = False
            if self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
        if self.adr_start:
            self.adr_start = False
            if self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
        # endregion Beginning Of New Session Management

        # region Trade Detection and Model Initialization
        # Only one trade per direction, the only case we have two trades would be a phase_4
        if self.session_dr_phase_1 and not self.session_dr_phase_2:
            if self.enable_phase_1_logging and not self.announce_p1:
                self.announce_p1 = True
                self.Log(
                    f"{self.current_session} Session DR range created. "
                    f"Price: {self.bar_close} should be within "
                    f"range [ {self.session_dr_high} -- {self.session_dr_low} ]"
                )
        elif self.session_dr_phase_2 and not self.session_dr_phase_3:
            if self.enable_logging and not self.announce_p2:
                self.announce_p2 = True
                self.Log(
                    f"Session Phase 2 in effect. {self.current_session} Session DR signal: {self.session_dr_direction}")
            if self.CheckEconomicImpact(self.Time):
                if self.session_dr_direction == 1 and not self.entry_models_initialized:
                    self.entry_models_initialized = True
                    if self.enable_phase_2_logging:
                        self.Log(f"Session Phase 2, Price: {self.bar_close} should be above {self.session_dr_high}.")
                    entry = self.session_idr_high - (self.session_idr_range * self.retracement_percent)
                    current_sl = self.session_idr_low - (self.session_idr_range * self.std_sl)
                    current_tp = self.session_idr_high + (self.session_idr_range * self.std_tp)
                    if self.VerifyRR(entry, current_sl, current_tp, self.session_dr_direction):
                        if self.enable_phase_2_logging:
                            self.Log(
                                f"First trade model initialized for session, Set Entry: {entry}, Sl: {current_sl}, Tp: {current_tp}")
                        self.session_long_entry_price = self.session_idr_high - (
                                    self.session_idr_range * self.retracement_percent)
                    else:
                        if self.enable_risk_verification_logging:
                            self.Log("RR Verification Failed, no trade taken!")
                elif self.session_dr_direction == -1 and not self.entry_models_initialized:
                    self.entry_models_initialized = True
                    if self.enable_phase_2_logging:
                        self.Log(f"Session Phase 2, Price: {self.bar_close} should be below {self.session_dr_low}.")
                    entry = self.session_idr_low + (self.session_idr_range * self.retracement_percent)
                    current_sl = self.session_idr_high + (self.session_idr_range * self.std_sl)
                    current_tp = self.session_idr_low - (self.session_idr_range * self.std_tp)
                    if self.VerifyRR(entry, current_sl, current_tp, self.session_dr_direction):
                        if self.enable_phase_2_logging:
                            self.Log(
                                f"First trade model initialized for session, Set Entry: {entry}, Sl: {current_sl}, Tp: {current_tp}")
                        self.session_short_entry_price = self.session_idr_low + (
                                    self.session_idr_range * self.retracement_percent)
                    else:
                        if self.enable_risk_verification_logging:
                            self.Log("RR Verification Failed, Trade was not executed!")
            else:
                if not self.announce_himpactnews:
                    self.announce_himpactnews = True
                    self.Log("There is Currently High Impact news this session, No trades shall be taken.")
                pass
        elif self.session_dr_phase_3 and not self.announce_p3:
            self.announce_p3 = True
            if self.enable_phase_3_logging:
                self.Log("Session has now become a false session")
        if self.session_dr_phase_4 and not self.announce_p4:
            self.announce_p4 = True
            if self.enable_phase_4_logging:
                self.Log("0.5 STD Reached!")
        # endregion Trade Detection and Model Initialization

        # region Trade Entry Management
        bar_max = max(self.bar_open, self.bar_high, self.bar_low, self.bar_close)
        bar_min = min(self.bar_open, self.bar_high, self.bar_low, self.bar_close)
        if self.session_dr_direction == 1 and self.session_long_entry_price != 0 and self.enable_long_entries:
            if self.can_trade and bar_min < self.session_long_entry_price < bar_max:
                self.can_trade = False
                self.total_entries += 1
                self.session_stoploss = self.session_idr_low - (self.session_idr_range * self.std_sl)
                self.session_takeprofit = self.session_idr_high + (self.session_idr_range * self.std_tp)
                order_size = self.ContractOrderSize(self.symbol, 1, self.session_long_entry_price,
                                                    self.session_stoploss)
                self.session_entry_ticket = self.MarketOrder(self.symbol, order_size)
                if self.enable_fill_logging:
                    self.Log(
                        f"{self.Time} Session Entry Ticket Filled, "
                        f"Setting sl: {self.session_stoploss} and tp: {self.session_takeprofit} "
                        f"sDirection: {self.session_dr_direction}, "
                        f"p2?: {self.session_dr_phase_2}, "
                        f"p3?: {self.session_dr_phase_3}"
                    )
        if self.session_dr_direction == -1 and self.session_short_entry_price != 0 and self.enable_short_entries:
            if self.can_trade and bar_min < self.session_short_entry_price < bar_max:
                self.can_trade = False
                self.total_entries += 1
                self.session_stoploss = self.session_idr_high + (self.session_idr_range * self.std_sl)
                self.session_takeprofit = self.session_idr_low - (self.session_idr_range * self.std_tp)
                order_size = self.ContractOrderSize(self.symbol, -1, self.session_short_entry_price,
                                                    self.session_stoploss)
                self.session_entry_ticket = self.MarketOrder(self.symbol, order_size)
                if self.enable_fill_logging:
                    self.Log(
                        f"{self.Time} Session Entry Ticket Filled, "
                        f"Setting sl: {self.session_stoploss} and tp: {self.session_takeprofit} "
                        f"sDirection: {self.session_dr_direction}, "
                        f"p2?: {self.session_dr_phase_2}, "
                        f"p3?: {self.session_dr_phase_3}"
                    )
        # endregion Trade Entry Management

        # region Trade TP SL Management
        if self.session_stoploss != 0 and self.session_takeprofit != 0:
            # Longs only, price below or equal to sl, close!
            if self.session_dr_direction == 1 and bar_min < self.session_stoploss < bar_max:
                self.Log("Long stoploss hit! Closing position and reseting trade parameters.")
                self.total_exits_sl += 1
                self.MarketOrder(self.symbol, (self.Portfolio[self.symbol].Quantity * -1), tag="lSl")
                self.ResetTradeParams()
            # Shorts only, price above or equal to sl, close!
            if self.session_dr_direction == -1 and bar_min < self.session_stoploss < bar_max:
                self.Log("Short stoploss hit! Closing position and reseting trade parameters.")
                self.total_exits_sl += 1
                self.MarketOrder(self.symbol, (self.Portfolio[self.symbol].Quantity * -1), tag="sSl")
                self.ResetTradeParams()
            # Longs only, price above or equal tp, close!
            if self.session_dr_direction == 1 and bar_min < self.session_takeprofit < bar_max:
                self.Log("Long takeprofit hit! Closing position and reseting trade parameters.")
                self.total_exits_tp += 1
                self.MarketOrder(self.symbol, (self.Portfolio[self.symbol].Quantity * -1), tag="lTp")
                self.ResetTradeParams()
            # Shorts only, price below or equal to tp, close!
            if self.session_dr_direction == -1 and bar_min < self.session_takeprofit < bar_max:
                self.Log("Short takeprofit hit! Closing position and reseting trade parameters.")
                self.total_exits_tp += 1
                self.MarketOrder(self.symbol, (self.Portfolio[self.symbol].Quantity * -1), tag="sTp")
                self.ResetTradeParams()
        # endregion Trade TP SL Management

        # region End of Session Position and Pending Order Management
        if self.odr_over:
            self.odr_over = False
            if not self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
                # Reset the session Trade Params
            self.ResetSessionTradeParams()
        if self.rdr_over:
            self.rdr_over = False
            if not self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
            # Reset the session Trade Params
            self.ResetSessionTradeParams()
        if self.adr_over:
            self.adr_over = False
            if not self.cancel_orders_on_open:
                # Check the current status of the Portfolio
                self.CheckOpenPosition(self.symbol, self.bar_close)
                # Cancel any pending orders not entered
                self.CancelPendingOrders(self.symbol)
            # Reset the session Trade Params
            self.ResetSessionTradeParams()
            # Calculate and reset the ODR RDR ADR daily session trade params
            self.EndOfDailyRange()
        # endregion End of Session Position and Pending Order Management
