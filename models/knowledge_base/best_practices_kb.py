from utils.data_loader import load_json_data, save_json_data

class BestPracticesKnowledgeBase:
    def __init__(self, kb_path):
        self.kb_path = kb_path
        self.kb = self.load_kb()

    def load_kb(self):
        return load_json_data(self.kb_path)

    def save_kb(self):
        save_json_data(self.kb, self.kb_path)

    def get_best_practices(self, category, technology=None):
        if technology:
            return self.kb.get(category, {}).get(technology, [])
        return self.kb.get(category, {})

    def add_best_practice(self, category, practice, technology=None):
        if technology:
            if category not in self.kb:
                self.kb[category] = {}
            if technology not in self.kb[category]:
                self.kb[category][technology] = []
            self.kb[category][technology].append(practice)
        else:
            if category not in self.kb:
                self.kb[category] = []
            self.kb[category].append(practice)
        self.save_kb()

    def remove_best_practice(self, category, practice, technology=None):
        if technology:
            if category in self.kb and technology in self.kb[category]:
                self.kb[category][technology].remove(practice)
                self.save_kb()
                return True
        else:
            if category in self.kb:
                self.kb[category].remove(practice)
                self.save_kb()
                return True
        return False

    def get_all_categories(self):
        return list(self.kb.keys())

    def get_technologies_for_category(self, category):
        return list(self.kb.get(category, {}).keys())