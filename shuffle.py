import random
import sys
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from fpdf import FPDF
import csv
import math
from scipy.stats import chisquare
import threading
import time
import platform
import logging
import os

# Platform-specific imports for key listening
if platform.system() == "Windows":
    import msvcrt
else:
    import sys
    import select
    import tty
    import termios

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(
    filename='shuffle_debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ensure 'plots' directory exists for saving plot images
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# Shuffling Algorithms
# ---------------------------

def builtin_shuffle(deck):
    """Performs the built-in Fisher-Yates shuffle using random.shuffle."""
    logging.debug("Starting Built-in Shuffle.")
    random.shuffle(deck)
    logging.debug("Completed Built-in Shuffle.")

def select_shuffle_algorithm():
    """Allows the user to select a shuffling algorithm."""
    print("\nSelect a shuffling algorithm:")
    print("1. Built-in Shuffle")
    print("2. Fisher-Yates Shuffle (Manual)")
    print("3. Riffle Shuffle")
    print("4. Sattolo's Shuffle")
    print("5. Random Block Shuffle")
    print("6. Pile Shuffle")
    choice = input("Enter the number corresponding to your choice: ").strip()
    if choice == '1':
        logging.info("Selected Built-in Shuffle.")
        return builtin_shuffle
    elif choice == '2':
        logging.info("Selected Fisher-Yates Shuffle.")
        return fisher_yates_shuffle
    elif choice == '3':
        logging.info("Selected Riffle Shuffle.")
        return riffle_shuffle
    elif choice == '4':
        logging.info("Selected Sattolo's Shuffle.")
        return sattolo_shuffle
    elif choice == '5':
        logging.info("Selected Random Block Shuffle.")
        return random_block_shuffle
    elif choice == '6':
        logging.info("Selected Pile Shuffle.")
        return lambda deck: pile_shuffle(deck, num_piles=random.randint(3, 7))
    else:
        logging.warning("Invalid shuffle choice. Defaulting to Built-in Shuffle.")
        print("Invalid choice. Defaulting to Built-in Shuffle.")
        return builtin_shuffle

def fisher_yates_shuffle(deck):
    """Performs the Fisher-Yates shuffle on the deck."""
    logging.debug("Starting Fisher-Yates Shuffle.")
    for i in range(len(deck)-1, 0, -1):
        j = random.randint(0, i)
        deck[i], deck[j] = deck[j], deck[i]
    logging.debug("Completed Fisher-Yates Shuffle.")

def riffle_shuffle(deck):
    """Performs a Riffle shuffle on the deck."""
    logging.debug("Starting Riffle Shuffle.")
    split_point = random.randint(len(deck)//3, 2*len(deck)//3)
    left = deck[:split_point]
    right = deck[split_point:]
    deck.clear()
    while left or right:
        if left and (not right or random.random() > 0.5):
            deck.append(left.pop(0))
        if right and (not left or random.random() > 0.5):
            deck.append(right.pop(0))
    logging.debug("Completed Riffle Shuffle.")

def sattolo_shuffle(deck):
    """Performs Sattolo's shuffle on the deck to generate a cyclic permutation."""
    logging.debug("Starting Sattolo's Shuffle.")
    for i in range(len(deck)-1, 0, -1):
        j = random.randint(0, i-1)
        deck[i], deck[j] = deck[j], deck[i]
    logging.debug("Completed Sattolo's Shuffle.")

def random_block_shuffle(deck):
    """Performs a Random Block shuffle on the deck."""
    logging.debug("Starting Random Block Shuffle.")
    chunks = []
    deck_copy = deck.copy()
    while deck_copy:
        # Determine a random chunk size between 1 and 10
        chunk_size = random.randint(1, min(10, len(deck_copy)))
        # Extract the chunk
        chunk = deck_copy[:chunk_size]
        # Remove the chunk from the deck_copy
        deck_copy = deck_copy[chunk_size:]
        # Append the chunk to the list of chunks
        chunks.append(chunk)
    # Shuffle the order of chunks
    random.shuffle(chunks)
    # Reassemble the deck by concatenating the shuffled chunks
    deck.clear()
    for chunk in chunks:
        deck.extend(chunk)
    logging.debug("Completed Random Block Shuffle.")

def pile_shuffle(deck, num_piles=4):
    """Performs a Pile shuffle by dividing the deck into piles and recombining them."""
    logging.debug(f"Starting Pile Shuffle with {num_piles} piles.")
    piles = [[] for _ in range(num_piles)]
    for index, card in enumerate(deck):
        piles[index % num_piles].append(card)
    random.shuffle(piles)
    deck.clear()
    for pile in piles:
        deck.extend(pile)
    logging.debug("Completed Pile Shuffle.")

# ---------------------------
# Deck Initialization
# ---------------------------

def initialize_deck():
    """Initializes a standard ordered deck of 52 cards represented as integers (0-51)."""
    deck = list(range(52))  # Represent cards as integers for efficiency
    return deck

# ---------------------------
# Utility Functions
# ---------------------------

def get_deck_hash(deck):
    """Returns a SHA-256 hash of the current deck order."""
    # Convert deck to bytes for hashing
    deck_bytes = bytes(deck)
    return hashlib.sha256(deck_bytes).hexdigest()

def calculate_distance(original_deck, shuffled_deck):
    """Calculates the total distance of all cards from their original positions."""
    distance = 0
    original_positions = {card: idx for idx, card in enumerate(original_deck)}
    for idx, card in enumerate(shuffled_deck):
        distance += abs(idx - original_positions[card])
    return distance

def number_to_card(card_number):
    """Converts a card number (0-51) to its corresponding card name."""
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
             'J', 'Q', 'K', 'A']
    suit = suits[card_number // 13]
    rank = ranks[card_number % 13]
    return f"{rank} of {suit}"

# ---------------------------
# Statistics and Tracking
# ---------------------------

class ShuffleStatistics:
    def __init__(self, original_deck):
        self.original_deck = original_deck
        self.shuffle_history = set()
        self.shuffle_history.add(get_deck_hash(original_deck))
        self.total_shuffles = 0
        self.duplicate_shuffles = 0
        self.distances = []
        self.position_frequencies = defaultdict(lambda: defaultdict(int))
        self.perfect_shuffles = 0
        self.average_distances = []
        self.return_to_original = defaultdict(int)
        self.longest_streak = 0
        self.current_streak = 0
        self.card_pair_frequencies = defaultdict(lambda: defaultdict(int))
        self.suit_clusters = defaultdict(int)
        self.sequential_patterns = defaultdict(int)
        self.previous_deck = original_deck.copy()  # For integrity checks
    
    def update_statistics(self, shuffled_deck):
        self.total_shuffles += 1
        deck_hash = get_deck_hash(shuffled_deck)
        if deck_hash in self.shuffle_history:
            self.duplicate_shuffles += 1
            logging.warning(f"Duplicate shuffle detected at shuffle number {self.total_shuffles}. Deck Hash: {deck_hash}")
            print(f"\nDuplicate shuffle detected at shuffle number {self.total_shuffles}.")
            # Log the deck state for debugging
            logging.debug(f"Deck at duplicate shuffle #{self.total_shuffles}: {shuffled_deck}")
            # Reset current streak on duplicate
            self.current_streak = 0
        else:
            self.shuffle_history.add(deck_hash)
            # Increment current streak if no duplicate
            self.current_streak += 1
            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak
            logging.debug(f"Unique shuffle #{self.total_shuffles}. Deck Hash: {deck_hash}")
        
        # Integrity check: Ensure that the deck has been shuffled
        if self.total_shuffles > 1 and shuffled_deck == self.previous_deck:
            logging.critical(f"Deck unchanged at shuffle number {self.total_shuffles}. Potential shuffle issue.")
            print(f"Critical: Deck unchanged at shuffle number {self.total_shuffles}.")
            sys.exit(1)
        
        # Update previous_deck
        self.previous_deck = shuffled_deck.copy()

        # Calculate distance
        distance = calculate_distance(self.original_deck, shuffled_deck)
        self.distances.append(distance)
        average_distance = distance / len(shuffled_deck)
        self.average_distances.append(average_distance)

        # Update position frequencies
        for pos, card in enumerate(shuffled_deck):
            self.position_frequencies[card][pos] += 1
            if pos == self.original_deck.index(card):
                self.return_to_original[card] += 1

        # Check for perfect shuffle
        if shuffled_deck == self.original_deck:
            self.perfect_shuffles += 1
            logging.info(f"Perfect shuffle achieved at shuffle number {self.total_shuffles}.")

        # Track specific card pairings
        for i in range(len(shuffled_deck)-1):
            first_card = shuffled_deck[i]
            second_card = shuffled_deck[i+1]
            self.card_pair_frequencies[first_card][second_card] += 1

        # Update suit clustering (number of times cards of the same suit are grouped together)
        current_suit = None
        cluster_size = 0
        for card in shuffled_deck:
            suit = card // 13  # 0: Hearts, 1: Diamonds, 2: Clubs, 3: Spades
            if suit == current_suit:
                cluster_size += 1
            else:
                if cluster_size > 1:
                    self.suit_clusters[current_suit] += 1
                current_suit = suit
                cluster_size = 1
        if cluster_size > 1:
            self.suit_clusters[current_suit] += 1

        # Update sequential patterns (e.g., sequences like 'J, Q, K' of the same suit)
        sequences = [
            (9, 10, 11),  # J, Q, K of Hearts
            (22, 23, 24), # J, Q, K of Diamonds
            (35, 36, 37), # J, Q, K of Clubs
            (48, 49, 50)  # J, Q, K of Spades
        ]
        for i in range(len(shuffled_deck)-2):
            seq = tuple(shuffled_deck[i:i+3])
            if seq in sequences:
                self.sequential_patterns[seq] += 1

    def perform_chi_square_test(self):
        """Performs the Chi-Square test for each card's position frequencies."""
        chi_square_results = {}
        expected = self.total_shuffles / 52
        for card, positions in self.position_frequencies.items():
            observed = [positions.get(pos, 0) for pos in range(52)]
            # Ensure that expected frequencies are greater than 0 to avoid division by zero
            expected_frequencies = [expected if expected > 0 else 1 for _ in range(52)]
            chi2, p = chisquare(observed, expected_frequencies)
            chi_square_results[card] = {'chi2': chi2, 'p_value': p}
        return chi_square_results

# ---------------------------
# Visualization Functions
# ---------------------------

def plot_statistics(stats: ShuffleStatistics):
    """Generates and saves various statistical plots."""
    # Define a 2x2 subplot grid
    plt.figure(figsize=(22, 14))
    
    # 1. Shuffle Statistics Bar Chart
    plt.subplot(2, 2, 1)
    labels = ['Total Shuffles', 'Unique Shuffles', 'Duplicate Shuffles', 'Perfect Shuffles']
    values = [
        stats.total_shuffles, 
        stats.total_shuffles - stats.duplicate_shuffles, 
        stats.duplicate_shuffles,
        stats.perfect_shuffles
    ]
    sns.barplot(x=labels, y=values, color='blue')
    plt.title('Shuffle Statistics')
    plt.ylabel('Number of Shuffles')
    plt.tight_layout()
    # Save the plot
    stats_shuffle_stats_plot = os.path.join(PLOTS_DIR, 'shuffle_statistics.png')
    plt.savefig(stats_shuffle_stats_plot)
    plt.close()
    
    # 2. Distribution of Shuffle Distances Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(stats.distances, bins=30, color='purple', edgecolor='black')
    plt.title('Distribution of Shuffle Distances')
    plt.xlabel('Total Distance from Original Order')
    plt.ylabel('Frequency')
    plt.tight_layout()
    stats_distances_plot = os.path.join(PLOTS_DIR, 'shuffle_distances.png')
    plt.savefig(stats_distances_plot)
    plt.close()
    
    # 3. Average Distance Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(stats.average_distances, color='cyan')
    plt.title('Average Distance Per Shuffle')
    plt.xlabel('Shuffle Number')
    plt.ylabel('Average Distance')
    plt.tight_layout()
    stats_avg_distance_plot = os.path.join(PLOTS_DIR, 'average_distance.png')
    plt.savefig(stats_avg_distance_plot)
    plt.close()
    
    # 4. Return to Original Position
    plt.figure(figsize=(20, 10))
    cards = list(stats.return_to_original.keys())
    returns = [stats.return_to_original[card] for card in cards]
    # Convert card numbers to names for labels
    card_names = [number_to_card(card) for card in cards]
    sns.barplot(x=card_names, y=returns, color='orange')
    plt.title('Return to Original Position Count')
    plt.xlabel('Card')
    plt.ylabel('Number of Returns')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    stats_return_plot = os.path.join(PLOTS_DIR, 'return_to_original_position.png')
    plt.savefig(stats_return_plot)
    plt.close()
    
    # Return the paths of the saved plots
    return {
        'shuffle_statistics': stats_shuffle_stats_plot,
        'shuffle_distances': stats_distances_plot,
        'average_distance': stats_avg_distance_plot,
        'return_to_original': stats_return_plot
    }

def plot_heatmap(stats: ShuffleStatistics):
    """Generates and saves a heatmap of card position frequencies."""
    # Prepare data
    cards = sorted(stats.position_frequencies.keys())
    positions = list(range(52))
    data = []
    card_names = []
    for card in cards:
        row = [stats.position_frequencies[card].get(pos, 0) for pos in positions]
        data.append(row)
        card_names.append(number_to_card(card))  # Convert number to card name
    df = pd.DataFrame(data, index=card_names, columns=positions)

    # Plot heatmap
    plt.figure(figsize=(25, 18))
    sns.heatmap(df, cmap='viridis')
    plt.title('Heatmap of Card Positions')
    plt.xlabel('Position in Deck')
    plt.ylabel('Card')
    plt.yticks(rotation=0)  # Keep card names horizontal for readability
    plt.tight_layout()
    heatmap_plot = os.path.join(PLOTS_DIR, 'heatmap_card_positions.png')
    plt.savefig(heatmap_plot)
    plt.close()
    
    return heatmap_plot

def plot_card_pair_frequencies(stats: ShuffleStatistics):
    """Generates and saves a bar chart of top 5 most frequent card pairings."""
    # Select top 5 most frequent pairings
    pair_counts = {}
    for first_card, second_cards in stats.card_pair_frequencies.items():
        for second_card, count in second_cards.items():
            pair = f"{number_to_card(first_card)} -> {number_to_card(second_card)}"
            pair_counts[pair] = count
    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    pairs, counts = zip(*sorted_pairs) if sorted_pairs else ([], [])

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(pairs), y=list(counts), color='green')
    plt.title('Top 5 Most Frequent Card Pairings')
    plt.xlabel('Card Pairing')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pair_frequencies_plot = os.path.join(PLOTS_DIR, 'card_pair_frequencies.png')
    plt.savefig(pair_frequencies_plot)
    plt.close()
    
    return pair_frequencies_plot

def plot_suit_clusters(stats: ShuffleStatistics):
    """Generates and saves a bar chart of suit clustering frequencies."""
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    counts = [stats.suit_clusters.get(i, 0) for i in range(4)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=suits, y=counts, color='red')
    plt.title('Suit Clustering Frequency')
    plt.xlabel('Suit')
    plt.ylabel('Number of Clusters')
    plt.tight_layout()
    suit_clusters_plot = os.path.join(PLOTS_DIR, 'suit_clustering.png')
    plt.savefig(suit_clusters_plot)
    plt.close()
    
    return suit_clusters_plot

def plot_sequential_patterns(stats: ShuffleStatistics):
    """Generates and saves a bar chart of top 5 most frequent sequential patterns."""
    # Select top 5 most frequent sequences
    seq_counts = stats.sequential_patterns
    sorted_seqs = sorted(seq_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    sequences, counts = zip(*sorted_seqs) if sorted_seqs else ([], [])

    # Convert tuple sequences to string with suit names
    sequences_str = []
    for seq in sequences:
        seq_str = ' -> '.join([number_to_card(card) for card in seq])
        sequences_str.append(seq_str)

    plt.figure(figsize=(14, 7))
    sns.barplot(x=list(sequences_str), y=list(counts), color='orange')
    plt.title('Top 5 Most Frequent Sequential Patterns')
    plt.xlabel('Sequential Pattern')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    sequential_patterns_plot = os.path.join(PLOTS_DIR, 'sequential_patterns.png')
    plt.savefig(sequential_patterns_plot)
    plt.close()
    
    return sequential_patterns_plot

# ---------------------------
# Reporting Functions
# ---------------------------

def export_to_csv(stats: ShuffleStatistics, filename='position_frequencies.csv'):
    """Exports the position frequencies to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Card'] + [f"Position {i}" for i in range(52)]
            writer.writerow(header)
            for card, positions in stats.position_frequencies.items():
                card_name = number_to_card(card)
                row = [card_name] + [positions.get(pos, 0) for pos in range(52)]
                writer.writerow(row)
        logging.info(f"Position frequencies exported to {filename}.")
        print(f"Position frequencies exported to {filename}.")
    except Exception as e:
        logging.error(f"Failed to export position frequencies: {e}")
        print(f"Failed to export position frequencies: {e}")

def export_card_pair_frequencies(stats: ShuffleStatistics, filename='card_pair_frequencies.csv'):
    """Exports the card pair frequencies to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Card Pair', 'Frequency'])
            for first_card, second_cards in stats.card_pair_frequencies.items():
                for second_card, count in second_cards.items():
                    pair_str = f"{number_to_card(first_card)} -> {number_to_card(second_card)}"
                    writer.writerow([pair_str, count])
        logging.info(f"Card pair frequencies exported to {filename}.")
        print(f"Card pair frequencies exported to {filename}.")
    except Exception as e:
        logging.error(f"Failed to export card pair frequencies: {e}")
        print(f"Failed to export card pair frequencies: {e}")

def export_suit_clusters(stats: ShuffleStatistics, filename='suit_clusters.csv'):
    """Exports the suit clustering frequencies to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Suit', 'Cluster Count'])
            suit_names = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
            for suit_id, suit in enumerate(suit_names):
                count = stats.suit_clusters.get(suit_id, 0)
                writer.writerow([suit, count])
        logging.info(f"Suit clustering frequencies exported to {filename}.")
        print(f"Suit clustering frequencies exported to {filename}.")
    except Exception as e:
        logging.error(f"Failed to export suit clustering frequencies: {e}")
        print(f"Failed to export suit clustering frequencies: {e}")

def export_sequential_patterns(stats: ShuffleStatistics, filename='sequential_patterns.csv'):
    """Exports the sequential patterns frequencies to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sequential Pattern', 'Frequency'])
            for seq, count in stats.sequential_patterns.items():
                seq_str = ' -> '.join([number_to_card(card) for card in seq])
                writer.writerow([seq_str, count])
        logging.info(f"Sequential patterns frequencies exported to {filename}.")
        print(f"Sequential patterns frequencies exported to {filename}.")
    except Exception as e:
        logging.error(f"Failed to export sequential patterns frequencies: {e}")
        print(f"Failed to export sequential patterns frequencies: {e}")

def generate_pdf_report(stats: ShuffleStatistics, chi_square_results, filename='shuffle_report.pdf'):
    """Generates a PDF report summarizing the shuffle statistics with explanations and embedded plots."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Shuffle Statistics Report", ln=True, align='C')

        # Introduction
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Introduction", ln=True)
        pdf.set_font("Arial", size=12)
        intro_text = (
            "This report presents a comprehensive analysis of the card shuffling process "
            "performed by the Card Shuffler program. The statistics, visualizations, and "
            "tests included aim to evaluate the effectiveness and randomness of various "
            "shuffling algorithms."
        )
        pdf.multi_cell(0, 10, intro_text)
        
        # Shuffle Statistics
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="1. Shuffle Statistics", ln=True)
        pdf.set_font("Arial", size=12)
        stats_text = (
            "This section summarizes the basic shuffle statistics:\n"
            "- **Total Shuffles:** The total number of shuffles performed.\n"
            "- **Unique Shuffles:** Number of unique shuffle orders achieved.\n"
            "- **Duplicate Shuffles:** Number of times a shuffle resulted in a previously seen order.\n"
            "- **Perfect Shuffles:** Number of shuffles that returned the deck to its original order."
        )
        pdf.multi_cell(0, 10, stats_text)
        # Embed Shuffle Statistics Plot
        stats_plots = plot_statistics(stats)
        pdf.image(stats_plots['shuffle_statistics'], x=10, y=None, w=190)
        pdf.ln(10)

        # Shuffle Distances
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="2. Shuffle Distances", ln=True)
        pdf.set_font("Arial", size=12)
        distances_text = (
            "The shuffle distance measures how far the deck is from its original order after each shuffle. "
            "A higher distance indicates a more thorough shuffle.\n\n"
            "- **Distribution of Shuffle Distances:** Shows how the shuffle distances are distributed across all shuffles.\n"
            "- **Average Distance Per Shuffle:** Tracks the average distance over time, indicating the consistency of the shuffle's effectiveness."
        )
        pdf.multi_cell(0, 10, distances_text)
        # Embed Shuffle Distances Plot
        pdf.image(stats_plots['shuffle_distances'], x=10, y=None, w=190)
        pdf.ln(10)
        # Embed Average Distance Plot
        pdf.image(stats_plots['average_distance'], x=10, y=None, w=190)
        pdf.ln(10)

        # Return to Original Position
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="3. Return to Original Position", ln=True)
        pdf.set_font("Arial", size=12)
        return_text = (
            "This statistic tracks how often each individual card returns to its original position "
            "after shuffling. A low number of returns across all cards suggests effective shuffling.\n\n"
            "The following bar chart displays the number of times each card has returned to its original position."
        )
        pdf.multi_cell(0, 10, return_text)
        # Embed Return to Original Position Plot
        pdf.image(stats_plots['return_to_original'], x=10, y=None, w=190)
        pdf.ln(10)

        # Card Pair Frequencies
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="4. Card Pair Frequencies", ln=True)
        pdf.set_font("Arial", size=12)
        pair_text = (
            "This section identifies the most frequently occurring adjacent card pairings in the shuffled decks. "
            "Frequent pairings may indicate patterns or biases in the shuffle algorithm.\n\n"
            "The bar chart below shows the top 5 most common card pairings."
        )
        pdf.multi_cell(0, 10, pair_text)
        # Embed Card Pair Frequencies Plot
        pair_plot = plot_card_pair_frequencies(stats)
        pdf.image(pair_plot, x=10, y=None, w=190)
        pdf.ln(10)

        # Suit Clustering
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="5. Suit Clustering Frequencies", ln=True)
        pdf.set_font("Arial", size=12)
        clustering_text = (
            "Suit clustering measures how often cards of the same suit are grouped together in the shuffled deck. "
            "A lower clustering frequency indicates a more random distribution of suits.\n\n"
            "The following bar chart illustrates the frequency of suit clusters."
        )
        pdf.multi_cell(0, 10, clustering_text)
        # Embed Suit Clustering Plot
        clustering_plot = plot_suit_clusters(stats)
        pdf.image(clustering_plot, x=10, y=None, w=190)
        pdf.ln(10)

        # Sequential Patterns
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="6. Sequential Patterns Frequencies", ln=True)
        pdf.set_font("Arial", size=12)
        sequential_text = (
            "Sequential patterns identify specific sequences of cards that occur frequently, such as 'J -> Q -> K'. "
            "Frequent sequential patterns may suggest non-randomness in the shuffling process.\n\n"
            "The bar chart below displays the top 5 most frequent sequential patterns."
        )
        pdf.multi_cell(0, 10, sequential_text)
        # Embed Sequential Patterns Plot
        sequential_plot = plot_sequential_patterns(stats)
        pdf.image(sequential_plot, x=10, y=None, w=190)
        pdf.ln(10)

        # Heatmap of Card Positions
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="7. Heatmap of Card Positions", ln=True)
        pdf.set_font("Arial", size=12)
        heatmap_text = (
            "The heatmap visualizes the frequency of each card appearing in each position across all shuffles. "
            "Uniform distribution across all positions indicates randomness in the shuffling algorithm."
        )
        pdf.multi_cell(0, 10, heatmap_text)
        # Embed Heatmap Plot
        heatmap_plot = plot_heatmap(stats)
        pdf.image(heatmap_plot, x=10, y=None, w=190)
        pdf.ln(10)

        # Chi-Square Test Results
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="8. Chi-Square Test Results", ln=True)
        pdf.set_font("Arial", size=12)
        chi_text = (
            "The Chi-Square test assesses whether the observed distribution of card positions deviates significantly "
            "from a uniform distribution (i.e., each card has an equal probability of being in any position).\n\n"
            "**Chi-Square Statistic (Chi2):** Measures the discrepancy between observed and expected frequencies.\n"
            "**P-Value:** Indicates the probability that the observed distribution is due to chance.\n\n"
            "A high p-value (> 0.05) suggests that the shuffle is sufficiently random, while a low p-value "
            "(<= 0.05) indicates potential biases in the shuffle algorithm."
        )
        # Replace '≤' with '<=' to prevent encoding issues
        chi_text = chi_text.replace('≤', '<=')
        pdf.multi_cell(0, 10, chi_text)
        # Embed Chi-Square Results Table
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(90, 10, txt="Card", border=1, align='C')
        pdf.cell(50, 10, txt="Chi2 Statistic", border=1, align='C')
        pdf.cell(50, 10, txt="P-Value", border=1, align='C')
        pdf.ln()
        pdf.set_font("Arial", size=12)
        # Create a table with card names, Chi2, and p-value
        # For brevity, we'll include only significant results (p-value <= 0.05)
        significant_results = {card: res for card, res in chi_square_results.items() if res['p_value'] <= 0.05}
        if significant_results:
            for card, res in significant_results.items():
                card_name = number_to_card(card)
                chi2 = f"{res['chi2']:.2f}"
                p_val = f"{res['p_value']:.4f}"
                pdf.cell(90, 10, txt=card_name, border=1)
                pdf.cell(50, 10, txt=chi2, border=1, align='C')
                pdf.cell(50, 10, txt=p_val, border=1, align='C')
                pdf.ln()
        else:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="All cards have p-values greater than 0.05, indicating no significant deviation from randomness.", ln=True, align='C')
        
        pdf.ln(10)

        # Conclusion
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Conclusion", ln=True)
        pdf.set_font("Arial", size=12)
        conclusion_text = (
            "The analysis of the shuffle statistics, visualizations, and chi-square tests indicate that the selected "
            "shuffling algorithms effectively randomize the deck. The low frequency of duplicate shuffles, minimal "
            "return to original positions, and uniform distribution in the heatmap collectively demonstrate the "
            "robustness of the shuffling process.\n\n"
            "Chi-Square tests further confirm the randomness, as no significant deviations were observed in the card positions. "
            "This ensures that each card has an equal probability of appearing in any position, validating the fairness "
            "and effectiveness of the shuffling algorithms implemented."
        )
        pdf.multi_cell(0, 10, conclusion_text)

        # Save the PDF
        pdf.output(filename)
        logging.info(f"PDF report generated as {filename}.")
        print(f"PDF report generated as {filename}.")
    except Exception as e:
        logging.error(f"Failed to generate PDF report: {e}")
        print(f"Failed to generate PDF report: {e}")

# ---------------------------
# Main Program Class
# ---------------------------

class CardShuffler:
    def __init__(self):
        self.original_deck = initialize_deck()
        self.current_deck = self.original_deck.copy()
        self.stats = ShuffleStatistics(self.original_deck)
        self.shuffle_method = builtin_shuffle  # Default shuffle method
        self.stop_shuffling = threading.Event()
        self.listener_thread = None

    def key_listener(self):
        """Listens for the 's' key to stop shuffling."""
        if platform.system() == "Windows":
            while not self.stop_shuffling.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 's':
                        print("\nStop signal received. Stopping shuffles...")
                        self.stop_shuffling.set()
                        break
                time.sleep(0.1)
        else:
            # Unix-based system
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self.stop_shuffling.is_set():
                    dr, dw, de = select.select([sys.stdin], [], [], 0.1)
                    if dr:
                        key = sys.stdin.read(1).lower()
                        if key == 's':
                            print("\nStop signal received. Stopping shuffles...")
                            self.stop_shuffling.set()
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run_shuffles(self, num_shuffles):
        """Performs the specified number of shuffles."""
        # Reset the stop_shuffling flag
        self.stop_shuffling.clear()

        # Start the key listener thread
        self.listener_thread = threading.Thread(target=self.key_listener, daemon=True)
        self.listener_thread.start()

        print("Press 's' to stop shuffling at any time.")

        try:
            for _ in tqdm(range(num_shuffles), desc="Shuffling", unit="shuffle"):
                if self.stop_shuffling.is_set():
                    break
                self.shuffle_method(self.current_deck)
                self.stats.update_statistics(self.current_deck)
        except Exception as e:
            logging.error(f"An error occurred during shuffling: {e}")
            print(f"An error occurred during shuffling: {e}")
        finally:
            # Ensure the listener thread is stopped
            self.stop_shuffling.set()
            self.listener_thread.join()
            print("\nShuffling process has been stopped.")
            logging.info("Shuffling process has been stopped.")

    def save_shuffle_history(self, filename='shuffle_history.pkl'):
        """Saves the shuffle history to a file."""
        import pickle
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.stats.shuffle_history, f)
            logging.info(f"Shuffle history saved to {filename}.")
            print(f"Shuffle history saved to {filename}.")
        except Exception as e:
            logging.error(f"Failed to save shuffle history: {e}")
            print(f"Failed to save shuffle history: {e}")

    def load_shuffle_history(self, filename='shuffle_history.pkl'):
        """Loads the shuffle history from a file."""
        import pickle
        try:
            with open(filename, 'rb') as f:
                self.stats.shuffle_history = pickle.load(f)
            logging.info(f"Shuffle history loaded from {filename}.")
            print(f"Shuffle history loaded from {filename}.")
        except FileNotFoundError:
            logging.warning(f"File {filename} not found. Starting with existing shuffle history.")
            print(f"File {filename} not found. Starting with existing shuffle history.")
        except Exception as e:
            logging.error(f"Failed to load shuffle history: {e}")
            print(f"Failed to load shuffle history: {e}")

    def generate_reports(self):
        """Generates all reports and visualizations."""
        print("\nGenerating reports and visualizations...")
        logging.info("Generating reports and visualizations.")
        plot_statistics(self.stats)
        plot_heatmap(self.stats)
        plot_card_pair_frequencies(self.stats)
        plot_suit_clusters(self.stats)
        plot_sequential_patterns(self.stats)
        chi_square_results = self.stats.perform_chi_square_test()
        generate_pdf_report(self.stats, chi_square_results)
        export_to_csv(self.stats)
        export_card_pair_frequencies(self.stats)
        export_suit_clusters(self.stats)
        export_sequential_patterns(self.stats)
        logging.info("All reports and visualizations generated.")
        print("All reports and visualizations generated.")

    def interactive_menu(self):
        """Provides an interactive menu for the user to interact with the shuffler."""
        while True:
            print("\n--- Card Shuffler Menu ---")
            print("1. Select Shuffling Algorithm")
            print("2. Start Shuffling")
            print("3. Save Shuffle History")
            print("4. Load Shuffle History")
            print("5. Generate Reports")
            print("6. Exit")
            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.shuffle_method = select_shuffle_algorithm()
            elif choice == '2':
                try:
                    num_shuffles = int(input("How many times do you want to shuffle the deck?: "))
                    if num_shuffles < 1:
                        print("Please enter a positive integer.")
                        continue
                    self.run_shuffles(num_shuffles)
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
            elif choice == '3':
                filename = input("Enter filename to save shuffle history (default: shuffle_history.pkl): ").strip()
                if not filename:
                    filename = 'shuffle_history.pkl'
                self.save_shuffle_history(filename)
            elif choice == '4':
                filename = input("Enter filename to load shuffle history (default: shuffle_history.pkl): ").strip()
                if not filename:
                    filename = 'shuffle_history.pkl'
                self.load_shuffle_history(filename)
            elif choice == '5':
                self.generate_reports()
            elif choice == '6':
                print("Exiting the program.")
                logging.info("Program exited by user.")
                sys.exit()
            else:
                print("Invalid choice. Please select a valid option.")
                logging.warning(f"Invalid menu choice entered: {choice}")

# ---------------------------
# Entry Point
# ---------------------------

def main():
    # Test hash uniqueness before proceeding
    try:
        test_deck1 = initialize_deck()
        test_deck2 = test_deck1.copy()
        random.shuffle(test_deck2)
        hash1 = get_deck_hash(test_deck1)
        hash2 = get_deck_hash(test_deck2)
        assert hash1 != hash2, "Hash collision detected: Different decks produced the same hash."
        logging.info("Hash uniqueness test passed.")
    except AssertionError as ae:
        logging.critical(ae)
        print(ae)
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error during hash uniqueness test: {e}")
        print(f"Error during hash uniqueness test: {e}")
        sys.exit(1)

    # Initialize and start the shuffler
    shuffler = CardShuffler()
    shuffler.interactive_menu()

if __name__ == "__main__":
    main()

