import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class CreditCardOffer:
    card_name: str
    bank: str
    annual_fee: float
    welcome_bonus: str
    rewards_rate: Dict[str, float]
    interest_rate: float
    income_requirement: float
    insurance_benefits: List[str]
    additional_perks: List[str]
    url: str
    last_updated: datetime

class CreditCardScraper:
    """Scrapes credit card information from various Canadian financial websites"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cards_data = []
        self._setup_selenium()

    def _setup_selenium(self):
        """Initialize Selenium WebDriver with Chrome options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)

    def _parse_annual_fee(self, fee_text: str) -> float:
        """Convert annual fee text to float value"""
        fee_text = fee_text.lower().strip()
        if 'no fee' in fee_text or 'free' in fee_text:
            return 0.0
        matches = re.findall(r'\$?(\d+(?:\.\d{2})?)', fee_text)
        return float(matches[0]) if matches else 0.0

    def _parse_income_requirement(self, income_text: str) -> float:
        """Convert income requirement text to float value"""
        matches = re.findall(r'\$?(\d+(?:,\d{3})*)', income_text)
        if matches:
            return float(matches[0].replace(',', ''))
        return 0.0

    def _parse_interest_rate(self, rate_text: str) -> float:
        """Convert interest rate text to float value"""
        matches = re.findall(r'(\d+(?:\.\d{2})?)\s*%', rate_text)
        return float(matches[0]) if matches else 0.0

    def scrape_ratehub(self):
        """Scrape credit card data from RateHub"""
        url = 'https://www.ratehub.ca/credit-cards/best-credit-cards'
        
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'credit-card-item'))
            )
            
            cards = self.driver.find_elements(By.CLASS_NAME, 'credit-card-item')
            
            for card in cards:
                try:
                    name = card.find_element(By.CLASS_NAME, 'card-name').text
                    bank = card.find_element(By.CLASS_NAME, 'card-issuer').text
                    annual_fee_text = card.find_element(By.CLASS_NAME, 'annual-fee').text
                    
                    # Get detailed card info by clicking on the card
                    card.click()
                    time.sleep(1)  # Wait for modal to load
                    
                    details = self.driver.find_element(By.CLASS_NAME, 'card-details')
                    welcome_bonus = details.find_element(By.CLASS_NAME, 'welcome-bonus').text
                    rewards = self._parse_rewards(details.find_element(By.CLASS_NAME, 'rewards-rate').text)
                    
                    card_offer = CreditCardOffer(
                        card_name=name,
                        bank=bank,
                        annual_fee=self._parse_annual_fee(annual_fee_text),
                        welcome_bonus=welcome_bonus,
                        rewards_rate=rewards,
                        interest_rate=self._parse_interest_rate(
                            details.find_element(By.CLASS_NAME, 'interest-rate').text
                        ),
                        income_requirement=self._parse_income_requirement(
                            details.find_element(By.CLASS_NAME, 'income-requirement').text
                        ),
                        insurance_benefits=self._parse_benefits(
                            details.find_element(By.CLASS_NAME, 'insurance').text
                        ),
                        additional_perks=self._parse_benefits(
                            details.find_element(By.CLASS_NAME, 'perks').text
                        ),
                        url=card.find_element(By.TAG_NAME, 'a').get_attribute('href'),
                        last_updated=datetime.now()
                    )
                    
                    self.cards_data.append(card_offer)
                    
                except Exception as e:
                    print(f"Error processing card: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping RateHub: {str(e)}")
        
        finally:
            self.driver.quit()

    def _parse_rewards(self, rewards_text: str) -> Dict[str, float]:
        """Parse rewards text into structured format"""
        rewards = {}
        patterns = [
            (r'(\d+(?:\.\d+)?%?)(?:\s+points|\s+cash back)?\s+on\s+(.+?)(?=\d|$)', 'percentage'),
            (r'(\d+)\s+points\s+per\s+\$\s+on\s+(.+?)(?=\d|$)', 'points')
        ]
        
        for pattern, reward_type in patterns:
            matches = re.finditer(pattern, rewards_text, re.IGNORECASE)
            for match in matches:
                rate, category = match.groups()
                category = category.strip().lower()
                rate = float(rate.strip('%')) if '%' in rate else float(rate)
                rewards[category] = rate
                
        return rewards

    def _parse_benefits(self, benefits_text: str) -> List[str]:
        """Parse benefits text into list of benefits"""
        return [benefit.strip() for benefit in benefits_text.split('\n') if benefit.strip()]

    def scrape_canadian_banks(self):
        """Scrape credit card data from major Canadian banks"""
        banks = {
            'TD': 'https://www.td.com/ca/en/personal-banking/products/credit-cards/',
            'RBC': 'https://www.rbcroyalbank.com/credit-cards/index.html',
            'CIBC': 'https://www.cibc.com/en/personal-banking/credit-cards.html',
            'Scotiabank': 'https://www.scotiabank.com/ca/en/personal/credit-cards.html',
            'BMO': 'https://www.bmo.com/main/personal/credit-cards/'
        }
        
        for bank_name, url in banks.items():
            try:
                print(f"Scraping {bank_name} credit cards...")
                
                # Get the page content
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find card elements (this will vary by bank)
                card_elements = self._find_card_elements(soup, bank_name)
                
                for card_element in card_elements:
                    try:
                        card_data = self._extract_card_data(card_element, bank_name)
                        if card_data:
                            self.cards_data.append(card_data)
                    except Exception as e:
                        print(f"Error processing {bank_name} card: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error scraping {bank_name}: {str(e)}")
    
    def _find_card_elements(self, soup: BeautifulSoup, bank_name: str) -> List:
        """Find credit card elements in the HTML based on bank"""
        if bank_name == 'TD':
            return soup.select('.td-card-comparison')
        elif bank_name == 'RBC':
            return soup.select('.rbc-card-product')
        elif bank_name == 'CIBC':
            return soup.select('.card-item')
        elif bank_name == 'Scotiabank':
            return soup.select('.scotia-credit-card')
        elif bank_name == 'BMO':
            return soup.select('.bmo-credit-card-item')
        return []
    
    def _extract_card_data(self, card_element: BeautifulSoup, bank_name: str) -> Optional[CreditCardOffer]:
        """Extract credit card data from HTML element based on bank"""
        try:
            # Extract common fields (implementation will vary by bank)
            if bank_name == 'TD':
                card_name = card_element.select_one('.card-title').text.strip()
                annual_fee_text = card_element.select_one('.annual-fee').text.strip()
                welcome_bonus = card_element.select_one('.welcome-offer').text.strip()
                rewards_text = card_element.select_one('.rewards').text.strip()
                interest_rate_text = card_element.select_one('.interest-rate').text.strip()
                income_req_text = card_element.select_one('.income-requirement').text.strip()
                benefits = [item.text.strip() for item in card_element.select('.benefit-item')]
                perks = [item.text.strip() for item in card_element.select('.perk-item')]
                url = "https://www.td.com" + card_element.select_one('a')['href']
            else:
                # Default extraction logic (simplified)
                card_name = card_element.select_one('h3, .card-name, .title').text.strip()
                annual_fee_text = "0"  # Default
                welcome_bonus = ""
                rewards_text = ""
                interest_rate_text = "19.99%"  # Default
                income_req_text = "0"
                benefits = []
                perks = []
                url = ""
                
                # Try to find URL
                link = card_element.select_one('a')
                if link and 'href' in link.attrs:
                    url = link['href']
                    if not url.startswith('http'):
                        url = f"https://{bank_name.lower()}.com{url}"
            
            # Create card offer object
            return CreditCardOffer(
                card_name=card_name,
                bank=bank_name,
                annual_fee=self._parse_annual_fee(annual_fee_text),
                welcome_bonus=welcome_bonus,
                rewards_rate=self._parse_rewards(rewards_text),
                interest_rate=self._parse_interest_rate(interest_rate_text),
                income_requirement=self._parse_income_requirement(income_req_text),
                insurance_benefits=benefits,
                additional_perks=perks,
                url=url,
                last_updated=datetime.now()
            )
        except Exception as e:
            print(f"Error extracting {bank_name} card data: {str(e)}")
            return None

class CreditCardAnalyzer:
    """Analyzes and ranks credit card offers based on various criteria"""
    
    def __init__(self, cards_data: List[CreditCardOffer]):
        self.cards_data = cards_data
        self.df = self._create_dataframe()
        
        # Import visualization libraries only when needed
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.visualization_available = True
        except ImportError:
            self.visualization_available = False
            print("Visualization libraries not available. Install matplotlib and seaborn for visualization features.")

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert card offers to pandas DataFrame"""
        data = []
        for card in self.cards_data:
            row = {
                'card_name': card.card_name,
                'bank': card.bank,
                'annual_fee': card.annual_fee,
                'welcome_bonus': card.welcome_bonus,
                'interest_rate': card.interest_rate,
                'income_requirement': card.income_requirement,
                'url': card.url,
                'last_updated': card.last_updated
            }
            
            # Add rewards rates
            for category, rate in card.rewards_rate.items():
                row[f'reward_{category}'] = rate
            
            # Add benefits as boolean flags
            all_benefits = card.insurance_benefits + card.additional_perks
            for benefit in all_benefits:
                row[f'has_{benefit.lower().replace(" ", "_")}'] = True
            
            data.append(row)
        
        return pd.DataFrame(data)

    def rank_cards(self, 
                  spending_profile: Dict[str, float],
                  preferences: Dict[str, float]) -> pd.DataFrame:
        """
        Rank credit cards based on spending profile and preferences
        
        Args:
            spending_profile: Annual spending in different categories
            preferences: Weights for different card features
        
        Returns:
            DataFrame with ranked cards and scores
        """
        df = self.df.copy()
        
        # Calculate rewards value
        annual_rewards = []
        for _, card in df.iterrows():
            reward_value = 0
            for category, amount in spending_profile.items():
                reward_col = f'reward_{category}'
                if reward_col in card:
                    reward_value += (amount * card[reward_col] / 100)
            annual_rewards.append(reward_value)
        
        df['annual_rewards'] = annual_rewards
        df['net_value'] = df['annual_rewards'] - df['annual_fee']
        
        # Calculate weighted scores for other features
        for feature, weight in preferences.items():
            if feature in df.columns:
                df[f'{feature}_score'] = df[feature] * weight
        
        # Calculate total score
        score_columns = [col for col in df.columns if col.endswith('_score')]
        df['total_score'] = df[score_columns].sum(axis=1) + df['net_value']
        
        # Rank cards
        return df.sort_values('total_score', ascending=False)

    def generate_report(self, ranked_cards: pd.DataFrame) -> str:
        """Generate a detailed report of card rankings"""
        report = "=== Credit Card Rankings Report ===\n\n"
        
        # Top 5 Overall Cards
        report += "Top 5 Overall Cards:\n"
        for idx, card in ranked_cards.head().iterrows():
            report += f"\n{idx + 1}. {card['card_name']} ({card['bank']})"
            report += f"\n   Net Annual Value: ${card['net_value']:.2f}"
            report += f"\n   Annual Fee: ${card['annual_fee']:.2f}"
            report += f"\n   More Info: {card['url']}\n"
        
        # Best Cards by Category
        categories = [col.replace('reward_', '') for col in ranked_cards.columns 
                     if col.startswith('reward_')]
        
        report += "\nBest Cards by Category:\n"
        for category in categories:
            top_card = ranked_cards.nlargest(1, f'reward_{category}').iloc[0]
            report += f"\n{category.title()}:"
            report += f"\n- {top_card['card_name']}"
            report += f"\n- Reward Rate: {top_card[f'reward_{category}']}%\n"
        
        return report
        
    def visualize_rankings(self, ranked_cards: pd.DataFrame, top_n: int = 5) -> None:
        """
        Visualize credit card rankings
        
        Args:
            ranked_cards: DataFrame with ranked cards
            top_n: Number of top cards to display
        """
        if not self.visualization_available:
            print("Visualization libraries not available. Install matplotlib and seaborn.")
            return
            
        # Get top N cards
        top_cards = ranked_cards.head(top_n)
        
        # Set up the figure
        fig, axes = self.plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credit Card Analysis', fontsize=16)
        
        # 1. Net Value Comparison
        ax1 = axes[0, 0]
        top_cards.sort_values('net_value').plot(
            kind='barh', 
            y='net_value', 
            x='card_name',
            ax=ax1,
            color='skyblue'
        )
        ax1.set_title('Net Annual Value')
        ax1.set_xlabel('Value ($)')
        
        # 2. Annual Fee vs. Rewards
        ax2 = axes[0, 1]
        ax2.scatter(
            top_cards['annual_fee'],
            top_cards['annual_rewards'],
            s=100,
            alpha=0.7
        )
        for i, card in top_cards.iterrows():
            ax2.annotate(
                card['card_name'],
                (card['annual_fee'], card['annual_rewards']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        ax2.set_title('Annual Fee vs. Rewards')
        ax2.set_xlabel('Annual Fee ($)')
        ax2.set_ylabel('Annual Rewards ($)')
        
        # 3. Rewards by Category
        ax3 = axes[1, 0]
        reward_cols = [col for col in top_cards.columns if col.startswith('reward_')]
        if reward_cols:
            reward_data = top_cards[['card_name'] + reward_cols].set_index('card_name')
            reward_data.columns = [col.replace('reward_', '') for col in reward_data.columns]
            reward_data.plot(kind='bar', ax=ax3)
            ax3.set_title('Rewards by Category')
            ax3.set_ylabel('Reward Rate (%)')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'No reward category data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 4. Welcome Bonus Comparison
        ax4 = axes[1, 1]
        # Extract numeric values from welcome bonus text using regex
        welcome_values = []
        for bonus in top_cards['welcome_bonus']:
            matches = re.findall(r'(\d+)', str(bonus))
            welcome_values.append(int(matches[0]) if matches else 0)
        
        welcome_df = pd.DataFrame({
            'card_name': top_cards['card_name'],
            'welcome_value': welcome_values
        })
        
        welcome_df.sort_values('welcome_value').plot(
            kind='barh',
            y='welcome_value',
            x='card_name',
            ax=ax4,
            color='lightgreen'
        )
        ax4.set_title('Welcome Bonus Value')
        ax4.set_xlabel('Estimated Value ($)')
        
        self.plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.plt.show()
        
    def export_interactive_visualization(self, ranked_cards: pd.DataFrame, filename: str = 'card_visualization.html') -> None:
        """
        Export interactive visualizations using Plotly
        
        Args:
            ranked_cards: DataFrame with ranked cards
            filename: Output HTML file name
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available. Install plotly for interactive visualizations.")
            return
            
        # Get top 10 cards
        top_cards = ranked_cards.head(10)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Net Annual Value", 
                "Annual Fee vs. Rewards",
                "Rewards by Category",
                "Welcome Bonus Comparison"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Net Value Comparison
        fig.add_trace(
            go.Bar(
                x=top_cards['net_value'],
                y=top_cards['card_name'],
                orientation='h',
                marker_color='rgba(0, 255, 255, 0.6)',
                name='Net Value'
            ),
            row=1, col=1
        )
        
        # 2. Annual Fee vs. Rewards
        fig.add_trace(
            go.Scatter(
                x=top_cards['annual_fee'],
                y=top_cards['annual_rewards'],
                mode='markers+text',
                text=top_cards['card_name'],
                textposition='top center',
                marker=dict(
                    size=12,
                    color='rgba(255, 0, 255, 0.7)',
                    line=dict(width=1, color='rgba(0, 0, 0, 0.5)')
                ),
                name='Cards'
            ),
            row=1, col=2
        )
        
        # 3. Rewards by Category for the top card
        top_card = top_cards.iloc[0]
        reward_cols = [col for col in top_cards.columns if col.startswith('reward_')]
        categories = [col.replace('reward_', '') for col in reward_cols]
        values = [top_card[col] if col in top_card else 0 for col in reward_cols]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color='rgba(255, 255, 0, 0.6)',
                name=f"{top_card['card_name']} Rewards"
            ),
            row=2, col=1
        )
        
        # 4. Welcome Bonus Comparison
        # Extract numeric values from welcome bonus text
        welcome_values = []
        for bonus in top_cards['welcome_bonus']:
            matches = re.findall(r'(\d+)', str(bonus))
            welcome_values.append(int(matches[0]) if matches else 0)
        
        fig.add_trace(
            go.Bar(
                x=welcome_values,
                y=top_cards['card_name'],
                orientation='h',
                marker_color='rgba(0, 255, 0, 0.6)',
                name='Welcome Bonus'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Credit Card Analysis Dashboard",
            height=800,
            showlegend=False,
            template="plotly_dark"
        )
        
        # Save to HTML file
        fig.write_html(filename)
        print(f"Interactive visualization saved to {filename}")

def main():
    # Initialize scraper and get data
    scraper = CreditCardScraper()
    
    print("=== Canadian Credit Card Analyzer ===\n")
    print("Scraping credit card data from RateHub...")
    scraper.scrape_ratehub()
    
    print("\nScraping credit card data from major Canadian banks...")
    scraper.scrape_canadian_banks()
    
    print(f"\nTotal cards collected: {len(scraper.cards_data)}")
    
    # Initialize analyzer
    analyzer = CreditCardAnalyzer(scraper.cards_data)
    
    # Example spending profile
    spending_profile = {
        'groceries': 6000,  # $6000/year on groceries
        'gas': 2400,        # $2400/year on gas
        'dining': 3600,     # $3600/year on dining
        'travel': 2000,     # $2000/year on travel
        'other': 12000      # $12000/year on other purchases
    }
    
    # Example preferences
    preferences = {
        'welcome_bonus': 0.3,
        'interest_rate': -0.2,  # Negative weight because lower is better
        'income_requirement': -0.1
    }
    
    print("\nRanking cards based on spending profile and preferences...")
    # Rank cards
    ranked_cards = analyzer.rank_cards(spending_profile, preferences)
    
    # Generate and print report
    report = analyzer.generate_report(ranked_cards)
    print(report)
    
    # Save results to CSV
    ranked_cards.to_csv('credit_card_rankings.csv', index=False)
    print("\nRankings saved to 'credit_card_rankings.csv'")
    
    # Create visualizations
    try:
        print("\nGenerating visualizations...")
        analyzer.visualize_rankings(ranked_cards, top_n=10)
        
        print("\nCreating interactive dashboard...")
        analyzer.export_interactive_visualization(ranked_cards, 'credit_card_dashboard.html')
        print("Interactive dashboard saved to 'credit_card_dashboard.html'")
    except Exception as e:
        print(f"\nError creating visualizations: {str(e)}")
        print("Visualization libraries may not be installed. Run 'pip install matplotlib seaborn plotly'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 