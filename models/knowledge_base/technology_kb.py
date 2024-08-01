from pymongo import MongoClient

class TechnologyKnowledgeBase:
    def __init__(self, mongodb_uri, db_name):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]

    def get_technology_info(self, technology_name):
        return self.db.technologies.find_one({"name": technology_name})

    def get_best_practices(self, category):
        return list(self.db.best_practices.find({"category": category}))

    def get_design_patterns(self, pattern_type):
        return list(self.db.design_patterns.find({"type": pattern_type}))

    def add_technology(self, technology_data):
        self.db.technologies.insert_one(technology_data)

    def add_best_practice(self, best_practice_data):
        self.db.best_practices.insert_one(best_practice_data)

    def add_design_pattern(self, design_pattern_data):
        self.db.design_patterns.insert_one(design_pattern_data)

    def update_technology(self, technology_name, update_data):
        self.db.technologies.update_one({"name": technology_name}, {"$set": update_data})

    def get_compatible_technologies(self, technology_name):
        tech = self.get_technology_info(technology_name)
        if tech and 'compatible_with' in tech:
            return [self.get_technology_info(t) for t in tech['compatible_with']]
        return []

    def get_technology_best_practices(self, technology_name):
        tech = self.get_technology_info(technology_name)
        if tech and 'best_practices' in tech:
            return [self.db.best_practices.find_one({"_id": bp_id}) for bp_id in tech['best_practices']]
        return []